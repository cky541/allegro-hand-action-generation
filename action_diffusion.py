"""
轻量级时序动作Diffusion模型
用PPO/AMP轨迹数据训练条件动作生成器
可直接在CPU/GPU上后台运行
"""
import os
import sys
import time
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
from datetime import datetime

# ============================================================
# 1. 数据集：从.npy轨迹构建 (obs, action) 对
# ============================================================
class TrajectoryActionDataset(Dataset):
    """
    从.npy轨迹文件构建 (obs, action) 训练对。
    支持单文件和多文件混合训练
    
    你的轨迹格式: [T, 124维] = [obs_t0, obs_t1, ..., obs_t299]
    action = obs_{t+1} - obs_t (取前action_dim维)
    """
    def __init__(self, data_path, obs_dim=124, action_dim=16, mode="obs_diff", multi_traj=False):
        """
        data_path: 单文件路径或目录路径(多轨迹模式)
        multi_traj: 如果True, data_path被视为目录,加载所有.npy文件
        """
        self.obs_list = []
        self.action_list = []
        
        if multi_traj:
            # 多轨迹模式: 加载目录下所有.npy文件
            if os.path.isdir(data_path):
                npy_files = sorted(glob.glob(os.path.join(data_path, "*.npy")))
            else:
                npy_files = sorted(glob.glob(data_path))
            
            # 过滤掉已生成的diffusion文件
            npy_files = [f for f in npy_files if "diffusion_gen" not in os.path.basename(f)]
            
            print(f"[Dataset] Multi-trajectory mode: found {len(npy_files)} files")
            for fpath in npy_files:
                raw = np.load(fpath)
                self._process_single_traj(raw, obs_dim, action_dim, mode)
                print(f"  Loaded {os.path.basename(fpath)}: {raw.shape}")
        else:
            # 单轨迹模式
            raw = np.load(data_path)
            print(f"[Dataset] Single trajectory: {data_path}, shape={raw.shape}")
            self._process_single_traj(raw, obs_dim, action_dim, mode)
        
        # 合并所有轨迹
        self.obs_all = torch.cat(self.obs_list, dim=0)       # [N, obs_dim]
        self.action_all = torch.cat(self.action_list, dim=0)  # [N, action_dim]
        
        print(f"[Dataset] Total samples: {len(self.obs_all)}, obs={obs_dim}, action={action_dim}")
    
    def _process_single_traj(self, raw, obs_dim, action_dim, mode):
        obs = torch.tensor(raw, dtype=torch.float32)
        
        if mode == "obs_diff":
            obs_input = obs[:-1]
            action_target = obs[1:, :action_dim] - obs[:-1, :action_dim]
        elif mode == "obs_next":
            obs_input = obs[:-1]
            action_target = obs[1:, :action_dim]
        
        self.obs_list.append(obs_input)
        self.action_list.append(action_target)
    
    def __len__(self):
        return len(self.obs_all)
    
    def __getitem__(self, idx):
        return self.obs_all[idx], self.action_all[idx]


# ============================================================
# 2. 噪声调度 (Cosine调度，更温和)
# ============================================================
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.02)


# ============================================================
# 3. 条件UNet — 轻量版（条件=观测状态）
# ============================================================
class ConditionalUNet(nn.Module):
    """
    条件去噪网络：输入噪声动作 + 观测条件 + 时间步 → 预测噪声
    
    架构：MLP-based（适合低维动作空间）
    - 输入: noisy_action(16) + obs_cond(124) + time_embed(32) = 172维
    - 输出: 预测噪声(16维)
    """
    def __init__(self, obs_dim=124, action_dim=16, hidden_dim=512, time_embed_dim=32):
        super().__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        
        # 时间步嵌入
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # 主网络
        total_input_dim = action_dim + obs_dim + time_embed_dim
        self.net = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
    
    def forward(self, x, t, cond):
        """
        x: [B, action_dim] 带噪声的动作
        t: [B] 时间步 (整数0~T-1)
        cond: [B, obs_dim] 观测条件
        """
        t = t.float().unsqueeze(-1) / 1000.0  # 归一化到[0,1]
        t_embed = self.time_mlp(t)              # [B, time_embed_dim]
        
        h = torch.cat([x, cond, t_embed], dim=-1)  # [B, action_dim+obs_dim+time_embed]
        return self.net(h)  # [B, action_dim]


# ============================================================
# 4. Diffusion模型
# ============================================================
class ActionDiffusion(nn.Module):
    """
    条件动作Diffusion模型
    - 训练: 给定(obs, action_gt)，加噪后预测噪声
    - 采样: 给定obs，从纯噪声逐步去噪得到动作
    """
    def __init__(self, obs_dim=124, action_dim=16, timesteps=100, 
                 beta_start=1e-4, beta_end=0.02, device="cpu"):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.timesteps = timesteps
        self.device = device
        
        # 噪声调度
        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        
        # 去噪网络
        self.denoise_net = ConditionalUNet(obs_dim, action_dim)
    
    def forward(self, x0, t, cond):
        """
        训练: 加噪后预测噪声
        x0: [B, action_dim] 真实动作
        t: [B] 时间步
        cond: [B, obs_dim] 观测条件
        """
        noise = torch.randn_like(x0)
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        
        # 前向加噪: x_t = sqrt(α̅_t) * x0 + sqrt(1-α̅_t) * ε
        x_t = sqrt_alpha_bar * x0 + sqrt_one_minus * noise
        
        # 预测噪声
        pred_noise = self.denoise_net(x_t, t, cond)
        return pred_noise, noise
    
    @torch.no_grad()
    def sample(self, cond, num_steps=None):
        """
        从噪声生成动作
        cond: [B, obs_dim] 观测条件
        """
        self.eval()
        if num_steps is None:
            num_steps = self.timesteps
        
        batch_size = cond.shape[0]
        x_t = torch.randn(batch_size, self.action_dim, device=self.device)
        
        for t_idx in reversed(range(num_steps)):
            t = torch.full((batch_size,), t_idx, device=self.device, dtype=torch.long)
            
            pred_noise = self.denoise_net(x_t, t, cond)
            
            # DDPM采样
            alpha = self.alphas[t_idx]
            alpha_bar = self.alphas_cumprod[t_idx]
            beta = self.betas[t_idx]
            
            if t_idx > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            
            x_t = (1 / torch.sqrt(alpha)) * (
                x_t - (beta / torch.sqrt(1 - alpha_bar)) * pred_noise
            ) + torch.sqrt(beta) * noise
        
        return x_t


# ============================================================
# 5. 训练函数
# ============================================================
def train_diffusion(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    # 数据
    dataset = TrajectoryActionDataset(
        args.data_path, 
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        mode=args.data_mode,
        multi_traj=args.multi_traj
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    # 模型
    model = ActionDiffusion(
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        timesteps=args.timesteps,
        device=device,
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 日志
    log_dir = os.path.join(args.log_dir, f"diffusion_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model params: {total_params:,}")
    print(f"[INFO] Logging to: {log_dir}")
    
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        
        for batch_idx, (obs, action) in enumerate(dataloader):
            obs = obs.to(device)
            action = action.to(device)
            
            batch_size = obs.shape[0]
            t = torch.randint(0, args.timesteps, (batch_size,), device=device)
            
            pred_noise, true_noise = model(action, t, obs)
            loss = F.mse_loss(pred_noise, true_noise)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            if global_step % args.log_interval == 0:
                writer.add_scalar("Loss/train", loss.item(), global_step)
                lr_now = optimizer.param_groups[0]['lr']
                print(f"[Step {global_step}] loss={loss.item():.6f} lr={lr_now:.2e}")
        
        scheduler.step()
        epoch_loss /= len(dataloader)
        epoch_time = time.time() - epoch_start
        
        # 每epoch采样可视化
        with torch.no_grad():
            sample_obs = obs[:4]
            gen_actions = model.sample(sample_obs)
            writer.add_histogram("generated_actions", gen_actions, epoch)
        
        print(f"[Epoch {epoch+1}/{args.epochs}] loss={epoch_loss:.6f} time={epoch_time:.1f}s")
        
        # 保存checkpoint
        if (epoch + 1) % args.save_interval == 0 or epoch_loss < best_loss:
            best_loss = min(best_loss, epoch_loss)
            ckpt_path = os.path.join(log_dir, f"model_epoch{epoch+1}_loss{epoch_loss:.4f}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, ckpt_path)
            print(f"  [SAVE] {ckpt_path}")
    
    # 保存最终模型
    final_path = os.path.join(log_dir, "model_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\n[DONE] Final model saved to {final_path}")
    writer.close()


# ============================================================
# 6. 推理/采样演示
# ============================================================
@torch.no_grad()
def demo_sampling(model_path, data_path, device="cuda"):
    """加载模型并采样演示"""
    model = ActionDiffusion(obs_dim=124, action_dim=16, timesteps=100, device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    raw = np.load(data_path)
    obs_data = torch.tensor(raw[:10, :124], dtype=torch.float32, device=device)
    
    gen_actions = model.sample(obs_data)
    
    print(f"Generated actions shape: {gen_actions.shape}")
    print(f"Action stats: mean={gen_actions.mean().item():.4f}, std={gen_actions.std().item():.4f}")
    
    return gen_actions.cpu().numpy()


# ============================================================
# 7. Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Action Diffusion Training")
    parser.add_argument("--data_path", type=str, 
                        default=r"D:\projects\data\oakink\processed\demo_traj_27500.npy",
                        help="Path to .npy trajectory file, or directory if multi_traj")
    parser.add_argument("--obs_dim", type=int, default=124)
    parser.add_argument("--action_dim", type=int, default=16,
                        help="Output action dimension. 16 for Allegro Hand (16 joints)")
    parser.add_argument("--data_mode", type=str, default="obs_diff", 
                        choices=["obs_diff", "obs_next"])
    parser.add_argument("--timesteps", type=int, default=100, help="Diffusion T")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=200)
    parser.add_argument("--log_dir", type=str, default=r"D:\projects\data\oakink\processed\logs")
    parser.add_argument("--multi_traj", action="store_true", default=False,
                        help="Enable multi-trajectory mode (data_path is directory)")
    parser.add_argument("--demo", action="store_true", help="Run demo sampling")
    parser.add_argument("--checkpoint", type=str, default=None, help="Load checkpoint for demo")
    
    args = parser.parse_args()
    
    if args.demo:
        if args.checkpoint is None:
            print("Please specify --checkpoint for demo mode")
            sys.exit(1)
        demo_sampling(args.checkpoint, args.data_path)
    else:
        train_diffusion(args)
