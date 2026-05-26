# allegro-hand-action-generation
From PPO to AMP to Diffusion MoE - Action Generation on Allegro Hand

基于 Isaac Lab 的 Allegro 灵巧手立方体重定向动作生成项目。

## 项目简介

本项目使用 Isaac Lab 仿真平台，在 Allegro Hand 机器人上实现立方体（DexCube）的**灵巧操作（In-Hand Manipulation）**任务，包括：

- **PPO 预训练**：使用 RSL-RL 训练 PPO 策略完成立方体重定向
- **AMP 运动风格迁移**：从预训练 PPO 轨迹中提取参考动作，通过对抗运动先验（AMP）学习指定风格的动作（如指尖触碰立方体）
- **轨迹数据处理与可视化**：生成、回放、筛选演示轨迹
- **模型对比分析**：多 checkpoint 数据对比、多视角视频录制

## 环境配置

| 项目 | 说明 |
|:----|:-----|
| 仿真平台 | Isaac Sim 5.1.0 / Isaac Lab |
| RL 框架 | RSL-RL |
| 机器人 | Allegro Hand (16-DOF) |
| 目标物体 | DexCube (立方体) |
| 任务类型 | 灵巧手内重定向 (In-Hand Repose) |
| GPU | NVIDIA GeForce RTX 4060 Laptop |

## 项目结构
├── PPO转换的用作AMP训练的轨迹/ # PPO生成的演示轨迹文件(.npy)
├── rsl_rl/ # RSL-RL 训练相关文件
│ ├── play.py # 运行已训练checkpoint
│ ├── gen_demo_traj_hydra.py # 生成演示轨迹
│ └── amp_train_from_ppo_data.py # AMP训练脚本
├── diffusion/ # Diffusion 模型相关代码
├── replay_traj.py # 回放.npy轨迹文件
├── compare_checkpoints.py # 多checkpoint数据对比
└── 图片视频文件/ # 训练曲线、结果截图
├── ppo train mean reward.png # PPO训练奖励曲线
├── ppo consecutive successes.png # PPO连续成功率曲线
├── ppo loss_entropy.png # PPO熵损失
├── ppo loss_value.png # PPO价值损失
├── diffusion_loss.png # Diffusion损失曲线
├── diffusion_loss多轨迹扩散.png # 多轨迹Diffusion损失
├── generated_actions.png # 生成动作可视化
├── generated action多轨迹扩散.png # 多轨迹生成动作可视化
├── MoE_router1 # MoE路由结果
├── MoE router2.png # MoE路由可视化
├── MoE_loss1.png # MoE损失曲线
├── ppomodel27500.mp4 # 生成动作可视化
├── generated action多轨迹扩散.png # 多轨迹生成动作可视化
├── MoE_router1 # MoE路由结果
├── MoE router2.png # MoE路由可视化
└── MoE_loss1.png # MoE损失曲线

## 实验结果

### PPO 训练结果

| 指标 | Step 22500 | Step 25500 | Step 27500 | Step 29000 | Step 31994 |
|:----|:----------:|:----------:|:----------:|:----------:|:----------:|
| Train/mean_reward | 215.31 | 230.03 | **243.16** | 203.51 | 211.49 |
| consecutive_successes | 0.593 | 0.367 | **0.637** | 0.606 | 0.610 |

**最佳 checkpoint**: Step 27500（最高奖励 + 最高成功率）

### 实验视频

实验结果视频已上传至百度网盘（待上传），包含：

| 视频文件 | 说明 |
|:--------|:-----|
| `ppomodel27500.mp4` | PPO model_27500 运行效果 |
| `AMPmodel27500.mp4` | AMP 训练结果 |
| `MoE初始结果.mp4` | MoE 模型初始结果 |
| `单轨迹diffusion.mp4` | 单轨迹 Diffusion 模型结果 |
| `多轨迹diffusion视频.mp4` | 多轨迹 Diffusion 模型结果 |

### 方法对比

| 方法 | 说明 | 状态 |
|:----|:-----|:----|
| **PPO** | 标准强化学习训练 | ✅ 已完成 |
| **AMP** | 从PPO轨迹中迁移动作风格 | ✅ 已完成 |
| **Diffusion (单轨迹)** | 扩散模型生成单条动作轨迹 | ✅ 已完成 |
| **Diffusion (多轨迹)** | 扩散模型生成多条动作轨迹 | ✅ 已完成 |
| **MoE** | 混合专家模型 | ✅ 初步完成 |

## 环境配置

| 项目 | 说明 |
|:----|:-----|
| 仿真平台 | Isaac Sim 5.1.0 / Isaac Lab |
| RL 框架 | RSL-RL |
| 机器人 | Allegro Hand (16-DOF) |
| 目标物体 | DexCube (立方体) |
| 任务类型 | 灵巧手内重定向 (In-Hand Repose) |
| GPU | NVIDIA GeForce RTX 4060 Laptop |

## 快速开始

### 1. 运行已训练的 PPO checkpoint

```bash
python rsl_rl/play.py \
    --task Isaac-Repose-Cube-Allegro-Direct-v0 \
    --num_envs 1 \
    --load_run "allegro_hand/训练文件夹" \
    --checkpoint "model_27500"
### 2. 回放演示轨迹
BASH
python replay_traj.py
### 3. AMP 训练
BASH
python rsl_rl/amp_train_from_ppo_data.py \
    --task Isaac-Repose-Cube-Allegro-Direct-v0 \
    --num_envs 128 \
    --max_iterations 5000 \
    --ref_data PPO转换的用作AMP训练的轨迹/demo_traj_27500.npy \
    --amp_beta 0.2 \
    --headless
## 参考
Isaac Lab
RSL-RL
Allegro Hand
License
BSD-3-Clause
