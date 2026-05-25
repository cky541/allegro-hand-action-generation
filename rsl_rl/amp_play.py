# scripts/reinforcement_learning/rsl_rl/amp_play.py
"""AMP trained model inference - bypasses optimizer loading issues."""
import argparse, sys, os, time
import torch
import gymnasium as gym
from isaaclab.app import AppLauncher
import cli_args

parser = argparse.ArgumentParser(description="AMP Play (skip optimizer)")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Repose-Cube-Allegro-Direct-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point",
                    help="Name of the RL agent configuration entry point.")
parser.add_argument("--real-time", action="store_true", default=False)
# --checkpoint 已由 cli_args.add_rsl_rl_args 包含，不需要重复定义
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata
installed_version = metadata.version("rsl-rl-lib")
from packaging import version

from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config
import isaaclab_tasks

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs or env_cfg.scene.num_envs
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = agent_cfg.device

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Create runner (gets the correct policy architecture)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    # Load checkpoint manually - only actor weights + normalizer
    resume_path = retrieve_file_path(args_cli.checkpoint)
    print(f"[INFO] Loading checkpoint from: {resume_path}")
    loaded_dict = torch.load(resume_path, map_location=agent_cfg.device)
    print(f"[INFO] Checkpoint keys: {list(loaded_dict.keys())}")
    
    # Directly load actor_state_dict (all we need for inference)
    runner.alg.actor.load_state_dict(loaded_dict["actor_state_dict"])
    
    # Try to load obs normalizer
    if "obs_normalizer_state_dict" in loaded_dict:
        runner.alg.actor.obs_normalizer.load_state_dict(loaded_dict["obs_normalizer_state_dict"])
        print("[INFO] Loaded observation normalizer")
    
    print("[INFO] Model weights loaded successfully!")
    
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    
    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
        if args_cli.real_time:
            sleep_time = dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
