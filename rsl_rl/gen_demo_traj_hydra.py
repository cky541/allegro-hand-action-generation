# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Generate demo trajectory using trained PPO checkpoint (Hydra-based)."""

import argparse
import sys
import os
import torch
import numpy as np
from isaaclab.app import AppLauncher
from omegaconf import OmegaConf
import yaml



# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Generate demo trajectory from trained checkpoint.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during inference.")
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (must be 1 for trajectory).")
parser.add_argument("--task", type=str, default="Isaac-Repose-Cube-Allegro-Direct-v0", help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent config entry point.")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=0, help="Not used, kept for compatibility.")
parser.add_argument("--distributed", action="store_true", default=False)
parser.add_argument("--export_io_descriptors", action="store_true", default=False)
parser.add_argument("--ray-proc-id", "-rid", type=int, default=None)
parser.add_argument("--ckpt_iter", type=int, default=31994, help="Checkpoint iteration number")
parser.add_argument("--ckpt_dir", type=str, 
    default=r"C:\Users\chong\isaaclab\logs\rsl_rl\allegro_hand\2026-05-07_20-19-39",
    help="Checkpoint directory")
parser.add_argument("--output_dir", type=str,
    default=r"D:\projects\data\oakink\processed",
    help="Output directory")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg, DirectMARLEnvCfg
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, handle_deprecated_rsl_rl_cfg
import isaaclab_tasks  # noqa: F401
from datetime import datetime
import logging
import importlib.metadata as metadata  # <-- 添加这一行

logger = logging.getLogger(__name__)

# ==================== 固定配置 ====================
# ==================== 配置 ====================
CKPT_DIR = args_cli.ckpt_dir
CKPT_FILE = os.path.join(CKPT_DIR, f"model_{args_cli.ckpt_iter}.pt")
SAVE_PATH = os.path.join(args_cli.output_dir, f"demo_traj_{args_cli.ckpt_iter}.npy")
MAX_STEPS = 2000
os.makedirs(args_cli.output_dir, exist_ok=True)
# ==============================================

# ==================================================

os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    # override configurations
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg.max_iterations = 0  # no training

    # handle deprecated configurations
    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # 注释掉手动配置，使用默认配置
    # env_cfg.obs_type = "full"
    # env_cfg.asymmetric_obs = False

    # set seed and device
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    print(f"观测空间: {env.observation_space}")
    print(f"观测维度: {env.observation_space.shape}")

    # create runner (only for loading checkpoint)
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_traj_gen"
    log_dir = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name, log_dir)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    # load checkpoint
    print(f"[INFO] Loading checkpoint from: {CKPT_FILE}")
    runner.load(CKPT_FILE)
    actor = runner.alg.actor
    actor.eval()
    print("✅ Checkpoint loaded.")

    # generate trajectory
    print(">>> Generating demo trajectory...")
    obs, _ = env.reset()
    print("obs keys:", obs.keys())
    for k, v in obs.items():
        print(f"  {k}: shape {v.shape}")

    obs_td = obs["policy"]  # 初始化 obs_td

    traj = []
    ep_reward = 0.0

    for step in range(MAX_STEPS):
        with torch.no_grad():
            # 包装成字典，键为 'policy'
            obs_dict = {"policy": obs_td}
            actions = actor(obs_dict).detach().view(env.num_envs, -1)
        obs, rewards, dones, _ = env.step(actions)  # env.step 返回 obs 字典
        obs_td = obs["policy"]  # 更新 obs_td
        traj.append(obs_td[0].cpu().numpy())
        ep_reward += rewards[0].item()
        if dones[0]:
            break

    np.save(SAVE_PATH, np.array(traj))
    print(f"🎉 Trajectory saved to {SAVE_PATH}, steps: {len(traj)}, dim: {traj[0].shape[0]}, reward: {ep_reward:.2f}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
