"""
MoE-Diffusion闭环验证 - 接入Isaac Sim
"""
import argparse
import sys
import os
import time
import torch
import numpy as np
import gymnasium as gym
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="MoE Diffusion Play")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="Isaac-Repose-Cube-Allegro-Direct-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--real-time", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata
from packaging import version
installed_version = metadata.version("rsl-rl-lib")

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config
import isaaclab_tasks

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from action_diffusion_moe import MoEActionDiffusion


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs or env_cfg.scene.num_envs
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = agent_cfg.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    device = agent_cfg.device

    # 加载MoE模型
    model = MoEActionDiffusion(obs_dim=124, action_dim=16, timesteps=100,
                                num_experts=4, top_k=2, device=device)
    ckpt = torch.load(args_cli.checkpoint, map_location=device)
    if "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    print(f"[INFO] MoE-Diffusion loaded from: {args_cli.checkpoint}")

    # 推理循环
    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    print("[INFO] Starting MoE-Diffusion control loop...")

    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            if isinstance(obs, torch.Tensor):
                obs_cond = obs.clone().detach()
            else:
                obs_cond = torch.as_tensor(obs['policy'], dtype=torch.float32, device=device)
            if obs_cond.dim() == 1:
                obs_cond = obs_cond.unsqueeze(0)
            obs_cond = obs_cond[:, :124]

            delta_pos = model.sample(obs_cond)
            current_pos = obs_cond[:, :16]
            target_pos = (current_pos + delta_pos).clamp(-1.0, 1.0)
            obs, _, dones, _ = env.step(target_pos)

        if args_cli.real_time:
            sleep_time = dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
