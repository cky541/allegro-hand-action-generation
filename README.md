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

