# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
AMP Training Script for Allegro Hand In-Hand Reorientation
Using Pre-trained PPO Trajectories as Reference Data

使用方法：
  python scripts\reinforcement_learning\rsl_rl\amp_train_from_ppo_data.py ^
      --task Isaac-Repose-Cube-Allegro-Direct-v0 ^
      --num_envs 128 ^
      --max_iterations 5000 ^
      --ref_data D:\projects\data\oakink\processed\demo_traj_27500.npy ^
      --obs_feature_dim 124 ^
      --amp_beta 0.2 ^
      --init_std 0.8 ^
      --std_decay 0.9995 ^
      --min_std 0.05 ^
      --headless

修复说明（v2）：
  [修复] std_decay 失效问题：将 std_param 从 optimizer 中移除，
         改为完全由外部手动调度衰减，不再受 optimizer 梯度更新干扰。
  [新增] 训练循环开始前检测并移除 std_param 的 optimizer 归属。
  [改进] 日志信息增加修复确认输出，方便调试。
  [新增] 每 1000 次迭代打印一次 std 衰减进度。
"""

import argparse
import sys
import os
import csv
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from isaaclab.app import AppLauncher
import cli_args

# ============================================================================
# 0. Argument Parsing
# ============================================================================
parser = argparse.ArgumentParser(description="AMP training with custom PPO reference data.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--distributed", action="store_true", default=False)
parser.add_argument("--export_io_descriptors", action="store_true", default=False)
parser.add_argument("--ray-proc-id", "-rid", type=int, default=None)

# --- AMP specific arguments ---
parser.add_argument("--amp_beta", type=float, default=0.2, help="AMP reward weight")
parser.add_argument("--amp_beta_schedule", type=str, default="constant", choices=["constant", "linear_warmup", "linear_decay"])
parser.add_argument("--disc_lr", type=float, default=1e-5, help="Discriminator learning rate")
parser.add_argument("--disc_batch_size", type=int, default=256)
parser.add_argument("--disc_update_freq", type=int, default=1)
parser.add_argument("--ref_data", type=str, default=r"D:\projects\data\oakink\processed\demo_traj_31994.npy")
parser.add_argument("--obs_feature_dim", type=int, default=124)
parser.add_argument("--r1_penalty_coeff", type=float, default=10.0)
parser.add_argument("--reward_type", type=str, default="negative_log", choices=["negative_log", "positive_log", "linear"])

# --- Action std decay arguments ---
parser.add_argument("--init_std", type=float, default=0.8, help="Initial action noise std")
parser.add_argument("--std_decay", type=float, default=0.9995, help="Std decay multiplier per iteration")
parser.add_argument("--min_std", type=float, default=0.05, help="Minimum action noise std")
parser.add_argument("--std_decay_interval", type=int, default=1, help="Decay every N iterations")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Version checks ---
import importlib.metadata as metadata
import platform
from packaging import version

RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(f"Please install correct RSL-RL version: {' '.join(cmd)}")
    exit(1)

import logging
import time
from datetime import datetime
from typing import Tuple
import gymnasium as gym
from tensordict import TensorDict
from rsl_rl.runners import OnPolicyRunner
from torch.utils.tensorboard import SummaryWriter

from isaaclab.envs import (
    DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg,
    ManagerBasedRLEnvCfg, multi_agent_to_single_agent,
)
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
import isaaclab_tasks
from isaaclab_tasks.utils.hydra import hydra_task_config

logger = logging.getLogger(__name__)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ============================================================================
# 1. AMP 判别器
# ============================================================================
class AMPDiscriminator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.8)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, s: torch.Tensor, s_next: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s, s_next], dim=-1))


# ============================================================================
# 2. 参考数据采样器
# ============================================================================
class ReferenceSampler:
    def __init__(self, filepath: str, device: torch.device, obs_dim: int):
        raw_data = np.load(filepath)
        print(f"[INFO] Loaded reference data: shape {raw_data.shape}")

        if raw_data.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {raw_data.shape}")

        if obs_dim < raw_data.shape[1]:
            raw_data = raw_data[:, :obs_dim]
            print(f"[INFO] Truncated to dim {obs_dim}")

        self.pairs_s = torch.tensor(raw_data[:-1], dtype=torch.float32, device=device)
        self.pairs_s_next = torch.tensor(raw_data[1:], dtype=torch.float32, device=device)
        self.num_pairs = len(self.pairs_s)
        print(f"[INFO] Created {self.num_pairs} (s, s_next) pairs")

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = min(batch_size, self.num_pairs)
        indices = torch.randint(0, self.num_pairs, (batch_size,), device=self.pairs_s.device)
        return self.pairs_s[indices], self.pairs_s_next[indices]


# ============================================================================
# 3. AMP Reward
# ============================================================================
def compute_amp_reward(scores: torch.Tensor, reward_type: str = "negative_log", eps: float = 1e-8) -> torch.Tensor:
    if reward_type == "negative_log":
        return -torch.log(scores + eps)
    elif reward_type == "positive_log":
        return -torch.log(1.0 - scores + eps)
    else:
        return scores


# ============================================================================
# 4. 手动训练循环（修复 std_decay 版本）
# ============================================================================
def manual_train_with_amp(
    env,
    alg,
    num_iterations,
    num_steps_per_env,
    amp_discriminator,
    amp_disc_optimizer,
    amp_ref_sampler,
    amp_obs_dim,
    amp_beta_fn,
    disc_update_freq,
    disc_batch_size,
    r1_penalty_coeff,
    reward_type,
    amp_device,
    log_dir,
    # --- std 衰减参数 ---
    init_std=0.8,
    std_decay=0.9995,
    min_std=0.05,
    std_decay_interval=1,
):
    num_envs = env.unwrapped.num_envs
    total_steps = 0

    amp_metrics = {
        'd_loss': 0.0, 'r1_penalty': 0.0,
        'real_score': 0.0, 'fake_score': 0.0,
        'mean_amp_r': 0.0, 'beta': 0.0,
        'action_std': 0.0,
    }

    # --- 设置日志和保存 ---
    model_dir = os.path.join(log_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    csv_path = os.path.join(log_dir, "amp_training_log.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["iteration", "action_std", "d_loss", "r1_penalty",
                         "real_score", "fake_score", "amp_reward", "beta",
                         "value_loss", "surrogate_loss", "entropy", "steps", "time_elapsed"])
    csv_file.flush()

    start_time = time.time()

    # ================================================================
    # [修复] 将 std_param 从 optimizer 中移除，改为手动调度
    # ================================================================
    std_param = None
    current_std = init_std
    
    if hasattr(alg.actor, 'distribution') and hasattr(alg.actor.distribution, 'std_param'):
        dist = alg.actor.distribution
        std_param = dist.std_param
        
        # 从 alg 的优化器中移除 std_param（找到 actor 对应的 optimizer）
        actor_optimizer = None
        if hasattr(alg, 'optimizer'):
            actor_optimizer = alg.optimizer
        elif hasattr(alg, 'actor_optimizer'):
            actor_optimizer = alg.actor_optimizer
        
        if actor_optimizer is not None:
            removed = False
            for param_group in actor_optimizer.param_groups:
                if std_param in param_group['params']:
                    param_group['params'].remove(std_param)
                    removed = True
            if removed:
                print(f"[FIX] ✓ Removed std_param from optimizer (will use manual schedule)")
            else:
                print(f"[WARN] std_param not found in optimizer param_groups")
        else:
            print(f"[WARN] Could not find actor optimizer!")
        
        # 设置初始 std
        with torch.no_grad():
            std_param.data.fill_(init_std)
        
        print(f"[INFO] Fixed std_param schedule: {init_std:.4f} → {min_std:.4f} "
              f"(decay={std_decay} per iter, interval={std_decay_interval})")
        print(f"[INFO] Warmup: first 200 iterations keep init_std={init_std}")
    else:
        print(f"[WARN] Could not find distribution.std_param! std decay disabled.")
    
    print(f"\n{'=' * 80}")
    print(f"Starting AMP manual training: {num_iterations} iterations")
    print(f"  num_envs={num_envs}, num_steps_per_env={num_steps_per_env}")
    print(f"  Init std={init_std}, Decay={std_decay}, Min std={min_std}")
    print(f"  Log dir: {log_dir}")
    print(f"{'=' * 80}\n")

    # 重置环境获取初始 obs
    obs_td, _ = env.reset()

    for iteration in range(num_iterations):
        iter_start = time.time()

        # ================================================================
        # Phase 1: 收集 rollout 数据
        # ================================================================
        alg.storage.clear()

        for step in range(num_steps_per_env):
            with torch.no_grad():
                actions = alg.act(obs_td)

            next_obs_td, rewards, dones, infos = env.step(actions)

            if not isinstance(rewards, torch.Tensor):
                rewards = torch.tensor(rewards, device=amp_device, dtype=torch.float32)
            if not isinstance(dones, torch.Tensor):
                dones = torch.tensor(dones, device=amp_device, dtype=torch.float32)

            alg.process_env_step(next_obs_td, rewards, dones, infos)

            obs_td = next_obs_td
            total_steps += num_envs

        # ================================================================
        # Phase 2: 注入 AMP reward
        # ================================================================
        storage = alg.storage
        T = storage.step

        if T >= 2 and amp_discriminator is not None:
            if 'policy' in storage.observations.keys():
                obs_all = storage.observations['policy']
            else:
                first_key = list(storage.observations.keys())[0]
                obs_all = storage.observations[first_key]

            raw_obs_dim = obs_all.shape[-1]
            use_dim = min(amp_obs_dim, raw_obs_dim)

            cur_obs = obs_all[:T-1, ..., :use_dim].reshape(-1, use_dim)
            nxt_obs = obs_all[1:T, ..., :use_dim].reshape(-1, use_dim)

            with torch.no_grad():
                scores = amp_discriminator(cur_obs, nxt_obs)
                amp_r_flat = compute_amp_reward(scores, reward_type=reward_type)
                amp_reward = amp_r_flat.view(T-1, num_envs, 1)

                beta = amp_beta_fn(iteration)
                storage.rewards[:T-1] += beta * amp_reward

                amp_metrics['mean_amp_r'] = (beta * amp_reward).mean().item()
                amp_metrics['beta'] = beta

            if iteration % disc_update_freq == 0:
                batch_size = min(disc_batch_size, cur_obs.shape[0])

                real_s, real_sn = amp_ref_sampler.sample(batch_size)

                idx = torch.randint(0, cur_obs.shape[0], (batch_size,), device=amp_device)
                fake_s = cur_obs[idx].detach()
                fake_sn = nxt_obs[idx].detach()

                real_score = amp_discriminator(real_s, real_sn)
                fake_score = amp_discriminator(fake_s, fake_sn)

                d_loss = -0.9 * torch.log(real_score + 1e-8).mean() - 0.9 * torch.log(1.0 - fake_score + 1e-8).mean()

                real_s.requires_grad_(True)
                real_sn.requires_grad_(True)
                real_score_r1 = amp_discriminator(real_s, real_sn)

                grad_s, grad_sn = torch.autograd.grad(
                    outputs=real_score_r1.sum(),
                    inputs=[real_s, real_sn],
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                )
                r1_penalty = (grad_s.pow(2).sum(dim=-1).mean() + grad_sn.pow(2).sum(dim=-1).mean()) * r1_penalty_coeff

                disc_loss = d_loss + r1_penalty

                amp_disc_optimizer.zero_grad()
                disc_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp_discriminator.parameters(), max_norm=1.0)
                amp_disc_optimizer.step()

                amp_metrics['d_loss'] = d_loss.item()
                amp_metrics['r1_penalty'] = r1_penalty.item()
                amp_metrics['real_score'] = real_score.mean().item()
                amp_metrics['fake_score'] = fake_score.mean().item()

        # ================================================================
        # Phase 3: compute_returns + PPO update
        # ================================================================
        alg.compute_returns(obs_td)
        loss_dict = alg.update()

        # ================================================================
        # Phase 4: [修复] action std 手动调度（完全绕过 optimizer）
        # ================================================================
        # std_param 已从 optimizer 中移除，现在完全由我们手动控制。
        # - 前 200 iter 保持 init_std 不变，让策略充分探索
        # - 之后每 std_decay_interval 次迭代执行一次衰减
        # ================================================================
        old_std = current_std

        if std_param is not None:
            if iteration >= 200 and iteration % std_decay_interval == 0:
                # 指数衰减
                current_std = current_std * std_decay
                current_std = max(current_std, min_std)
                
                with torch.no_grad():
                    std_param.data.fill_(current_std)
            
            # 记录实际 std（从分布中读取，验证一致性）
            actual_std = std_param.mean().item()
        else:
            actual_std = current_std

        amp_metrics['action_std'] = actual_std

        # ================================================================
        # 日志 + TensorBoard + CSV + 模型保存
        # ================================================================
        writer.add_scalar("AMP/D_loss", amp_metrics['d_loss'], iteration)
        writer.add_scalar("AMP/R1_penalty", amp_metrics['r1_penalty'], iteration)
        writer.add_scalar("AMP/Real_score", amp_metrics['real_score'], iteration)
        writer.add_scalar("AMP/Fake_score", amp_metrics['fake_score'], iteration)
        writer.add_scalar("AMP/AMP_reward", amp_metrics['mean_amp_r'], iteration)
        writer.add_scalar("AMP/Beta", amp_metrics['beta'], iteration)
        writer.add_scalar("PPO/Value_loss", loss_dict.get('value', 0), iteration)
        writer.add_scalar("PPO/Surrogate_loss", loss_dict.get('surrogate', 0), iteration)
        writer.add_scalar("PPO/Entropy", loss_dict.get('entropy', 0), iteration)
        writer.add_scalar("Train/Action_std", amp_metrics['action_std'], iteration)
        writer.add_scalar("Train/Steps", total_steps, iteration)

        elapsed = time.time() - start_time
        csv_writer.writerow([iteration, amp_metrics['action_std'], amp_metrics['d_loss'],
                             amp_metrics['r1_penalty'], amp_metrics['real_score'],
                             amp_metrics['fake_score'], amp_metrics['mean_amp_r'],
                             amp_metrics['beta'], loss_dict.get('value', 0),
                             loss_dict.get('surrogate', 0), loss_dict.get('entropy', 0),
                             total_steps, elapsed])
        csv_file.flush()

        # 每 100 次迭代保存模型
        if iteration > 0 and iteration % 100 == 0:
            save_dict = alg.save()
            save_dict['iteration'] = iteration
            torch.save(save_dict, os.path.join(model_dir, f"model_{iteration}.pt"))

        # 控制台日志（每 20 次打印一次）
        if iteration % 20 == 0:
            iter_time = time.time() - iter_start
            eta = (elapsed / (iteration + 1)) * (num_iterations - iteration - 1)

            decay_mark = ""
            if iteration >= 200 and iteration % std_decay_interval == 0 and iteration > 200:
                decay_mark = f"({old_std:.4f}->{actual_std:.4f})"
            
            # 每 1000 次打印详细的衰减总结
            if iteration % 1000 == 0 and iteration > 0:
                expected_std = init_std * (std_decay ** (iteration - 199))
                expected_std = max(expected_std, min_std)
                print(f"  [STD] Iter={iteration}: actual={actual_std:.4f}, expected≈{expected_std:.4f}, "
                      f"decayed_by={init_std / max(actual_std, 1e-8):.2f}x")
            
            print(f"[Iter {iteration:5d}/{num_iterations}] "
                  f"Steps={total_steps} | "
                  f"ActionStd={amp_metrics['action_std']:.4f}{decay_mark} | "
                  f"D_loss={amp_metrics['d_loss']:.4f} | "
                  f"Real={amp_metrics['real_score']:.3f} | "
                  f"Fake={amp_metrics['fake_score']:.3f} | "
                  f"AMP_R={amp_metrics['mean_amp_r']:.4f} | "
                  f"Beta={amp_metrics['beta']:.4f} | "
                  f"VL={loss_dict.get('value', 0):.2f} | "
                  f"SL={loss_dict.get('surrogate', 0):.4f} | "
                  f"T={iter_time:.2f}s | "
                  f"ETA={eta:.0f}s")

    # ================================================================
    # 训练结束：保存最终模型 + 关闭资源
    # ================================================================
    final_dict = alg.save()
    final_dict['iteration'] = num_iterations - 1
    final_dict['total_steps'] = total_steps
    torch.save(final_dict, os.path.join(log_dir, "model_final.pt"))
    
    torch.save({
        "model_state_dict": final_dict,
        "iteration": num_iterations - 1,
        "total_steps": total_steps,
    }, os.path.join(log_dir, f"model_{num_iterations}.pt"))
    
    print(f"[SAVE] Final model saved to {log_dir}/model_{num_iterations}.pt")

    writer.close()
    csv_file.close()

    total_time = time.time() - start_time
    print(f"\n{'=' * 80}")
    print(f"AMP Training completed! Total time: {total_time:.2f} seconds")
    print(f"Total steps: {total_steps}")
    print(f"Results saved to: {log_dir}")
    print(f"{'=' * 80}\n")


# ============================================================================
# 5. Main
# ============================================================================
@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
         agent_cfg: RslRlBaseRunnerCfg):
    """AMP Training with manual loop"""

    # --- 配置 ---
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # --- 日志目录 ---
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name + "_AMP"))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    env_cfg.log_dir = log_dir

    # --- 创建环境 ---
    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # --- 获取维度 ---
    device = env.unwrapped.device
    raw_obs_dim = env.observation_space.shape[-1]
    amp_obs_dim = min(args_cli.obs_feature_dim, raw_obs_dim)
    print(f"[INFO] Raw observation dim: {raw_obs_dim}, AMP observation dim: {amp_obs_dim}")

    # --- 初始化 AMP 组件 ---
    discriminator = AMPDiscriminator(input_dim=amp_obs_dim).to(device)
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=args_cli.disc_lr)
    ref_sampler = ReferenceSampler(filepath=args_cli.ref_data, device=device, obs_dim=amp_obs_dim)

    def amp_beta_fn(iteration):
        if args_cli.amp_beta_schedule == "constant":
            return args_cli.amp_beta
        elif args_cli.amp_beta_schedule == "linear_warmup":
            return args_cli.amp_beta * min(iteration / (0.5 * agent_cfg.max_iterations), 1.0)
        else:
            return args_cli.amp_beta * max(1.0 - iteration / agent_cfg.max_iterations, 0.0)

    # --- 创建标准 OnPolicyRunner ---
    start_time = time.time()
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)

    alg = runner.alg

    # ================================================================
    # 初始化后打印 distribution 调试信息
    # ================================================================
    print("\n" + "=" * 60)
    print("=== DISTRIBUTION DEBUG ===")
    dummy_obs = env.observation_space.sample()
    dummy_obs_td = TensorDict({"policy": torch.from_numpy(dummy_obs).unsqueeze(0).to(device)}, batch_size=[1])
    with torch.no_grad():
        actions = alg.act(dummy_obs_td)
    
    if hasattr(alg.actor, 'log_std'):
        print(f"alg.actor.log_std is nn.Parameter? {isinstance(alg.actor.log_std, nn.Parameter)}")
        print(f"alg.actor.log_std.shape: {alg.actor.log_std.shape}")
        print(f"alg.actor.log_std.data: {alg.actor.log_std.data}")
        print(f"std from log_std: {alg.actor.log_std.exp().mean().item():.6f}")
    else:
        print("alg.actor.log_std NOT FOUND")
    
    std_attrs = []
    for attr_name in dir(alg.actor):
        if 'std' in attr_name.lower() or 'log' in attr_name.lower():
            if not attr_name.startswith('_'):
                obj = getattr(alg.actor, attr_name)
                if isinstance(obj, (torch.Tensor, nn.Parameter)):
                    std_attrs.append(f"  actor.{attr_name}: {type(obj).__name__}, shape={obj.shape}, val={obj.mean().item():.4f}")
                else:
                    std_attrs.append(f"  actor.{attr_name}: {type(obj).__name__}")
    
    if std_attrs:
        print("std-related attributes on actor:")
        for s in std_attrs:
            print(s)
    else:
        print("No std-related attributes found on actor!")

    if hasattr(alg.actor, 'distribution'):
        dist = alg.actor.distribution
        print(f"\ndistribution type: {type(dist)}")
        print(f"distribution.__dict__ keys: {list(dist.__dict__.keys())}")
        for k, v in dist.__dict__.items():
            if isinstance(v, torch.Tensor):
                print(f"  dist.{k}: shape={v.shape}, val={v.mean().item():.6f}")
            else:
                print(f"  dist.{k}: {v}")
    else:
        print("alg.actor.distribution NOT FOUND")
    
    # 初始设定 std
    if hasattr(alg.actor, 'distribution') and hasattr(alg.actor.distribution, 'std_param'):
        with torch.no_grad():
            alg.actor.distribution.std_param.data.fill_(args_cli.init_std)
            print(f"[INFO] Set init std_param to {args_cli.init_std}")
            print(f"[INFO] Verified std = {alg.actor.distribution.std_param.mean().item():.6f}")
    else:
        print(f"[WARN] Could not find distribution.std_param to set init std!")

    print("=" * 60 + "\n")

    num_steps_per_env = alg.storage.num_transitions_per_env
    print(f"[INFO] num_steps_per_env = {num_steps_per_env}")

    # --- 保存配置 ---
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # --- 运行手动训练循环 ---
    manual_train_with_amp(
        env=env,
        alg=alg,
        num_iterations=agent_cfg.max_iterations,
        num_steps_per_env=num_steps_per_env,
        amp_discriminator=discriminator,
        amp_disc_optimizer=disc_optimizer,
        amp_ref_sampler=ref_sampler,
        amp_obs_dim=amp_obs_dim,
        amp_beta_fn=amp_beta_fn,
        disc_update_freq=args_cli.disc_update_freq,
        disc_batch_size=args_cli.disc_batch_size,
        r1_penalty_coeff=args_cli.r1_penalty_coeff,
        reward_type=args_cli.reward_type,
        amp_device=device,
        log_dir=log_dir,
        init_std=args_cli.init_std,
        std_decay=args_cli.std_decay,
        min_std=args_cli.min_std,
        std_decay_interval=args_cli.std_decay_interval,
    )

    total_time = time.time() - start_time
    print(f"\n{'=' * 80}")
    print(f"Total training time: {total_time:.2f} seconds")
    print(f"Results saved to: {log_dir}")
    print(f"{'=' * 80}\n")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
