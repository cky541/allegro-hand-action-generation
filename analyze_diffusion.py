"""分析Diffusion模型生成的动作质量"""
import torch
import numpy as np
import sys
import os

# 把 diffusion 目录加入路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from action_diffusion import ActionDiffusion

CHECKPOINT = r"D:\projects\data\oakink\processed\logs\diffusion_2026-05-24_00-06-42\model_final.pt"
DATA_PATH = r"D:\projects\data\oakink\processed\demo_traj_27500.npy"
OUTPUT_PATH = r"D:\projects\data\oakink\processed\diffusion_gen_full.npy"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

# 加载模型
model = ActionDiffusion(obs_dim=124, action_dim=16, timesteps=100, device=device)
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.to(device)
model.eval()
print("[INFO] Model loaded")

# 加载完整轨迹
raw = np.load(DATA_PATH)
print(f"[INFO] Trajectory shape: {raw.shape}")

# 采样全部
obs_all = torch.tensor(raw[:-1, :124], dtype=torch.float32, device=device)
with torch.no_grad():
    gen_all = model.sample(obs_all).cpu().numpy()

# 真实动作（相邻obs差）
true_all = raw[1:, :16] - raw[:-1, :16]

# ===== 统计对比 =====
print("\n" + "="*60)
print(f"真实动作: mean={true_all.mean():.6f}, std={true_all.std():.6f}")
print(f"生成动作: mean={gen_all.mean():.6f}, std={gen_all.std():.6f}")
print(f"MSE:      {((gen_all - true_all)**2).mean():.6f}")
print(f"MAE:      {np.abs(gen_all - true_all).mean():.6f}")
print(f"Max error: {np.abs(gen_all - true_all).max():.6f}")

# 逐关节相关性
corrs = []
for i in range(16):
    corr = np.corrcoef(gen_all[:, i], true_all[:, i])[0, 1]
    corrs.append(corr)
    print(f"  关节 {i:2d}: 相关系数 = {corr:.4f}")

print(f"\n平均相关系数: {np.mean(corrs):.4f}")
print(f"最小相关系数: {np.min(corrs):.4f} (关节 {np.argmin(corrs)})")
print(f"最大相关系数: {np.max(corrs):.4f} (关节 {np.argmax(corrs)})")

# 直方图分布对比
print("\n" + "="*60)
print("动作值分布对比 (百分位数):")
for q in [5, 25, 50, 75, 95]:
    t_q = np.percentile(true_all, q)
    g_q = np.percentile(gen_all, q)
    print(f"  P{q:2d}: 真实={t_q:.4f}, 生成={g_q:.4f}, 差异={g_q - t_q:.4f}")

# 保存
np.save(OUTPUT_PATH, gen_all)
print(f"\n[SAVED] {OUTPUT_PATH}")
print("="*60)
