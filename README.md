
```markdown
# Allegro Hand Action Generation

基于 Isaac Lab 的 Allegro 灵巧手立方体重定向动作生成项目。

## 项目简介

本项目在 Allegro Hand 16-DOF 灵巧手上实现立方体（DexCube）的**手内重定向（In-Hand Repose）**任务，探索多种动作生成方法：

- **PPO 强化学习**：使用 RSL-RL 训练 PPO 策略完成立方体重定向
- **AMP 运动风格迁移**：从 PPO 演示轨迹中提取参考动作，通过对抗运动先验学习固定风格
- **Diffusion 动作生成**：基于扩散模型生成灵巧手动作轨迹
- **MoE 混合专家**：探索多专家模型在灵巧操作中的应用

## 仓库结构

```
├── PPO转换的用作AMP训练的轨迹/      # PPO生成的演示轨迹文件(.npy)
├── rsl_rl/                          # RSL-RL 训练相关代码
│   ├── play.py                      # 运行已训练checkpoint
│   ├── gen_demo_traj_hydra.py       # 生成演示轨迹
│   └── amp_train_from_ppo_data.py   # AMP训练脚本
├── diffusion/                       # Diffusion 模型相关代码
├── replay_traj.py                   # 回放.npy轨迹文件
├── compare_checkpoints.py           # 多checkpoint数据对比
│
├── 实验视频/
│   ├── ppomode27500.mp4             # PPO model_27500 运行效果（前/右/上三视角）
│   ├── AMPmodel27500.mp4            # AMP 训练结果
│   ├── MoE初始结果.mp4              # MoE 模型初始结果
│   ├── 单轨迹diffusion.mp4          # 单轨迹 Diffusion 模型结果
│   └── 多轨迹diffusion视频.mp4       # 多轨迹 Diffusion 模型结果
│
├── 训练曲线/
│   ├── ppo train_mean reward.png     # PPO训练奖励曲线
│   ├── ppo consecutive_successes.png # PPO连续成功率曲线
│   ├── ppo loss_entropy.png          # PPO熵损失
│   ├── ppo loss_value.png            # PPO价值损失
│   ├── diffusion_loss.png            # Diffusion损失曲线
│   ├── diffusion_loss多轨迹扩散.png  # 多轨迹Diffusion损失
│   ├── generated_actions.png         # 单轨迹生成动作可视化
│   ├── generated action多轨迹扩散.png # 多轨迹生成动作可视化
│   ├── MoE_router1                   # MoE路由权重可视化
│   ├── MoE_router2.png               # MoE路由权重可视化
│   └── MoE_loss1.png                 # MoE损失曲线
└── README.md
```

## 实验结果

### PPO 训练结果

PPO 训练了 32000 步（8192 并行环境），从中筛选出综合表现最佳的 checkpoint。

| 指标 | Step 22500 | Step 25500 | Step 27500 | Step 29000 | Step 31994 |
|:----|:----------:|:----------:|:----------:|:----------:|:----------:|
| Train/mean_reward | 215.31 | 230.03 | **243.16** | 203.51 | 211.49 |
| consecutive_successes | 0.593 | 0.367 | **0.637** | 0.606 | 0.610 |

**最佳 checkpoint**: Step 27500（最高奖励 + 最高成功率）

### 各方法效果对比

| 方法 | 效果 |
|:----|:-----|
| **PPO** ✅ | 在固定初始条件下有**较高的重定向成功率**，策略能稳定完成立方体旋转 |
| **AMP** ⚠️ | 从 PPO 轨迹中学到了**固定的动作模式**，能够复现特定的指尖触碰风格 |
| **Diffusion** ⚠️ | 成功生成了动作轨迹，但质量和多样性有待提升 |
| **MoE** ⚠️ | 初步实现了多专家路由，但专家分工尚不明确 |

### 当前局限

虽然 PPO 在**固定初始位姿和目标位姿**下表现出不错的重定向成功率，但：

> ❌ 面对**随机目标位姿**和**随机初始位姿**时，重定向效果并不理想。
>
> 模型学到的更多是**固定的动作序列**，而非通用的重定向策略，泛化能力不足。

### 改善思路

针对上述问题，目前的改进方向是：

1. **任务分解**：将重定向任务拆解为绕 **X 轴、Y 轴、Z 轴**的三个独立旋转动作轨迹
2. **MoE + 判别器**：为每个旋转轴训练一个 Expert，并引入判别器来选择/组合合适的专家
3. **目标泛化**：通过专家分工，使模型能够根据不同的目标位姿自适应地选择旋转策略，提升对随机目标的泛化能力

```
重定向任务
    │
    ├── Expert X (绕X轴旋转) ─── 判别器
    ├── Expert Y (绕Y轴旋转) ─── 判别器  ─── 组合输出 → 重定向动作
    └── Expert Z (绕Z轴旋转) ─── 判别器
```

## 实验视频

| 视频 | 说明 |
|:----|:-----|
| [`ppomode27500.mp4`](./ppomode27500.mp4) | PPO model_27500 运行效果（前/右/上三视角） |
| [`AMPmodel27500.mp4`](./AMPmodel27500.mp4) | AMP 训练结果，从 PPO 轨迹迁移风格 |
| [`MoE初始结果.mp4`](./MoE初始结果.mp4) | MoE 模型初始运行结果 |
| [`单轨迹diffusion.mp4`](./单轨迹diffusion.mp4) | 单轨迹 Diffusion 模型生成结果 |
| [`多轨迹diffusion视频.mp4`](./多轨迹diffusion视频.mp4) | 多轨迹 Diffusion 模型生成结果 |

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
    --load_run "allegro_hand/2026-05-07_20-19-39" \
    --checkpoint "model_27500"
```

### 2. 回放演示轨迹

```bash
python replay_traj.py
```

### 3. AMP 训练

```bash
python rsl_rl/amp_train_from_ppo_data.py \
    --task Isaac-Repose-Cube-Allegro-Direct-v0 \
    --num_envs 128 \
    --max_iterations 5000 \
    --ref_data PPO转换的用作AMP训练的轨迹/demo_traj_27500.npy \
    --amp_beta 0.2 \
    --headless
```

## 参考

- [Isaac Lab](https://github.com/isaac-sim/IsaacLab)
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl)
- [Allegro Hand](https://www.wonikrobotics.com/research/allegro-hand)

## License

BSD-3-Clause
```
