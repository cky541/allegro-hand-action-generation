"""
Diffusion模型闭环验证 - 接入Isaac Sim
将Diffusion生成的动作序列喂入仿真环境，观察机械手行为
"""
import argparse
import sys
import os
import time
import torch
import numpy as np
import gymnasium as gym
from isaaclab.app import AppLauncher

# ===== 命令行参数 =====
parser = argparse.ArgumentParser(description="Diffusion Play in Isaac Sim")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="Isaac-Repose-Cube-Allegro-Direct-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point",
                    help="Name of the RL agent configuration entry point.")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to diffusion model checkpoint")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=300)
parser.add_argument("--real-time", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ===== 导入 =====
import importlib.metadata as metadata
from packaging import version
installed_version = metadata.version("rsl-rl-lib")

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab.utils.dict import print_dict
import isaaclab_tasks

# 导入Diffusion模型
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from action_diffusion import ActionDiffusion


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    # 手动覆盖参数，不再依赖 cli_args
    env_cfg.scene.num_envs = args_cli.num_envs or env_cfg.scene.num_envs
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = agent_cfg.device

    # 创建环境
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # 视频录制
    if args_cli.video:
        video_kwargs = {
            "video_folder": r"D:\projects\data\oakink\processed\videos",
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during playback.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ===== 加载Diffusion模型 =====
    device = agent_cfg.device
    diffusion = ActionDiffusion(obs_dim=124, action_dim=16, timesteps=100, device=device)
    ckpt = torch.load(args_cli.checkpoint, map_location=device)
    if "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    diffusion.load_state_dict(ckpt)
    diffusion.to(device)
    diffusion.eval()
    print(f"[INFO] Diffusion model loaded from: {args_cli.checkpoint}")

    # ===== 推理loop =====
    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    timestep = 0
    
    print("[INFO] Starting diffusion-based control loop...")
    
    while simulation_app.is_running():
        start_time = time.time()
        
        with torch.inference_mode():
            # IMPORTANT: RslRlVecEnvWrapper返回的obs可能是TensorDict
            # 确保转成普通Tensor，避免torch.cat时TensorDict拦截出错
            if isinstance(obs, torch.Tensor):
                obs_cond = obs.clone().detach()
            else:
                # TensorDict类型 - 用__getitem__提取tensor
                if hasattr(obs, 'get'):
                    obs_cond = torch.as_tensor(obs['policy'], dtype=torch.float32, device=device)
                else:
                    obs_cond = torch.as_tensor(obs, dtype=torch.float32, device=device)
            
            # 确保shape正确 [num_envs, 124]
            if obs_cond.dim() == 1:
                obs_cond = obs_cond.unsqueeze(0)
            obs_cond = obs_cond[:, :124]
            
            # Diffusion采样: 预测关节位置增量 delta_pos
            delta_pos = diffusion.sample(obs_cond)  # [B, 16]
            
            # 从obs中提取当前关节位置 (obs[:16] = 关节位置)
            current_pos = obs_cond[:, :16]
            
            # 目标关节位置 = 当前关节位置 + Diffusion预测增量
            target_pos = (current_pos + delta_pos).clamp(-1.0, 1.0)
            
            # 环境接受16维关节目标位置作为action
            actions = target_pos
            
            # 环境step
            obs, _, dones, _ = env.step(actions)

        # 视频录制控制
        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        # 实时控制
        if args_cli.real_time:
            sleep_time = dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
