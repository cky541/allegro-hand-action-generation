"""
MoE-Diffusion: 混合专家条件动作生成模型
用多个Expert分别建模不同动作模式，Router根据obs动态选择
面试亮点: MoE架构 + Diffusion生成 + 机器人动作建模
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
# 1. 数据集（复用多轨迹版本）
# ============================================================
class TrajectoryActionDataset(Dataset):
    def __init__(self, data_path, obs_dim=124, action_dim=16, mode="obs_diff", multi_traj=False):
        self.obs_list = []
        self.action_list = []
        
        if multi_traj:
            if os.path.isdir(data_path):
                npy_files = sorted(glob.glob(os.path.join(data_path, "*.npy")))
            else:
                npy_files = sorted(glob.glob(data_path))
            npy_files = [f for f in npy_files if "diffusion_gen" not in os.path.basename(f)]
            print(f"[Dataset] Multi-trajectory MoE: found {len(npy_files)} files")
            for fpath in npy_files:
                raw = np.load(fpath)
                self._process_single_traj(raw, obs_dim, action_dim, mode)
                print(f"  Loaded {os.path.basename(fpath)}: {raw.shape}")
        else:
            raw = np.load(data_path)
            print(f"[Dataset] Single trajectory: {data_path}, shape={raw.shape}")
            self._process_single_traj(raw, obs_dim, action_dim, mode)
        
        self.obs_all = torch.cat(self.obs_list, dim=0)
        self.action_all = torch.cat(self.action_list, dim=0)
        print(f"[Dataset] Total MoE samples: {len(self.obs_all)}, obs={obs_dim}, action={action_dim}")
    
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
# 2. 噪声调度
# ============================================================
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.02)


# ============================================================
# 3. 单个Expert网络
# ============================================================
class ExpertNet(nn.Module):
    """单个专家网络 - 处理特定动作模式"""
    def __init__(self, obs_dim=124, action_dim=16, hidden_dim=256, expert_id=0):
        super().__init__()
        self.expert_id = expert_id
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
        )
        total_input_dim = action_dim + obs_dim + 32
        self.net = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )
    
    def forward(self, x, t, cond):
        t = t.float().unsqueeze(-1) / 1000.0
        t_embed = self.time_mlp(t)
        h = torch.cat([x, cond, t_embed], dim=-1)
        return self.net(h)


# ============================================================
# 4. MoE去噪网络
# ============================================================
class MoEConditionalUNet(nn.Module):
    """
    MoE (Mixture of Experts) 条件去噪网络
    
    架构:
    - Router: 基于obs条件，输出每个Expert的权重
    - Experts: 多个独立ExpertNet，每个擅长不同动作模式
    - 输出: 加权融合所有Expert的输出
    
    面试亮点: 
    - 稀疏激活 (只选Top-K Expert)
    - 负载均衡 (辅助损失)
    - 条件路由 (基于观测状态选择Expert)
    """
    def __init__(self, obs_dim=124, action_dim=16, num_experts=4, hidden_dim=256, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.action_dim = action_dim
        
        # 路由网络: 基于obs输出每个Expert的权重
        self.router = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, num_experts),
        )
        
        # 多个专家网络
        self.experts = nn.ModuleList([
            ExpertNet(obs_dim, action_dim, hidden_dim, expert_id=i)
            for i in range(num_experts)
        ])
        
        # 初始化路由权重
        self._init_router()
    
    def _init_router(self):
        for m in self.router.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)
    
    def forward(self, x, t, cond, return_router_weights=False):
        """
        x: [B, action_dim] 带噪声的动作
        t: [B] 时间步
        cond: [B, obs_dim] 观测条件
        
        返回: 
        - output: [B, action_dim] 预测噪声
        - router_weights (可选): [B, num_experts] 路由权重
        - aux_loss: 负载均衡辅助损失
        """
        batch_size = x.shape[0]
        
        # 1. 路由: 基于cond计算每个Expert的权重
        router_logits = self.router(cond)  # [B, num_experts]
        router_weights = F.softmax(router_logits, dim=-1)  # [B, num_experts]
        
        # 2. Top-K稀疏激活
        top_k_weights, top_k_indices = torch.topk(router_weights, self.top_k, dim=-1)
        # 归一化Top-K权重
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # 3. 计算每个Expert的输出
        expert_outputs = torch.zeros(batch_size, self.action_dim, device=x.device)
        
        for k in range(self.top_k):
            expert_idx = top_k_indices[:, k]  # [B]
            weight_k = top_k_weights[:, k:k+1]  # [B, 1]
            
            # 对每个样本选择对应的expert
            for e_idx in range(self.num_experts):
                mask = (expert_idx == e_idx)
                if mask.any():
                    expert_out = self.experts[e_idx](x[mask], t[mask], cond[mask])
                    expert_outputs[mask] += weight_k[mask] * expert_out
        
        # 4. 负载均衡辅助损失 (鼓励均匀使用Expert)
        # 理想分布: 每个Expert被选中的概率 = 1/num_experts
        # 实际分布: router_weights.mean(dim=0)
        router_probs = router_weights.mean(dim=0)  # [num_experts]
        ideal_probs = torch.ones(self.num_experts, device=x.device) / self.num_experts
        aux_loss = F.kl_div(router_probs.log(), ideal_probs, reduction='batchmean')
        
        if return_router_weights:
            return expert_outputs, router_weights, aux_loss
        return expert_outputs, aux_loss


# ============================================================
# 5. MoE Diffusion模型
# ============================================================
class MoEActionDiffusion(nn.Module):
    """
    MoE条件动作Diffusion模型
    - 用多个Expert分别建模不同动作模式
    - Router基于obs动态选择Expert组合
    """
    def __init__(self, obs_dim=124, action_dim=16, timesteps=100, 
                 num_experts=4, top_k=2, device="cpu"):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.timesteps = timesteps
        self.num_experts = num_experts
        self.top_k = top_k
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
        
        # MoE去噪网络
        self.denoise_net = MoEConditionalUNet(obs_dim, action_dim, num_experts, top_k=top_k)
    
    def forward(self, x0, t, cond):
        """
        训练: 加噪后预测噪声 + 负载均衡损失
        """
        noise = torch.randn_like(x0)
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        
        x_t = sqrt_alpha_bar * x0 + sqrt_one_minus * noise
        pred_noise, aux_loss = self.denoise_net(x_t, t, cond)
        
        return pred_noise, noise, aux_loss
    
    @torch.no_grad()
    def sample(self, cond, num_steps=None):
        """从噪声生成动作 - 使用MoE网络"""
        self.eval()
        if num_steps is None:
            num_steps = self.timesteps
        
        batch_size = cond.shape[0]
        x_t = torch.randn(batch_size, self.action_dim, device=self.device)
        
        for t_idx in reversed(range(num_steps)):
            t = torch.full((batch_size,), t_idx, device=self.device, dtype=torch.long)
            
            pred_noise, _, _ = self.denoise_net(x_t, t, cond, return_router_weights=True)
            
            alpha = self.alphas[t_idx]
            alpha_bar = self.alphas_cumprod[t_idx]
            beta = self.betas[t_idx]
            
            noise = torch.randn_like(x_t) if t_idx > 0 else 0
            
            x_t = (1 / torch.sqrt(alpha)) * (
                x_t - (beta / torch.sqrt(1 - alpha_bar)) * pred_noise
            ) + torch.sqrt(beta) * noise
        
        return x_t
    
    @torch.no_grad()
    def analyze_routing(self, cond):
        """分析路由分布 - 用于可视化每个Expert的激活情况"""
        self.eval()
        router_logits = self.denoise_net.router(cond)
        router_weights = F.softmax(router_logits, dim=-1)
        return router_weights


# ============================================================
# 6. 训练函数
# ============================================================
def train_moe_diffusion(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] MoE config: {args.num_experts} experts, top-{args.top_k}")
    
    dataset = TrajectoryActionDataset(
        args.data_path, obs_dim=args.obs_dim, action_dim=args.action_dim,
        mode=args.data_mode, multi_traj=args.multi_traj
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    model = MoEActionDiffusion(
        obs_dim=args.obs_dim, action_dim=args.action_dim,
        timesteps=args.timesteps, num_experts=args.num_experts,
        top_k=args.top_k, device=device,
    ).to(device)
    
    # 区分参数组：diffusion损失和路由损失使用不同权重
    base_params = []
    router_params = []
    for name, param in model.named_parameters():
        if 'router' in name:
            router_params.append(param)
        else:
            base_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': base_params, 'lr': args.lr},
        {'params': router_params, 'lr': args.lr * 2.0},  # 路由网络学习率更高
    ], weight_decay=1e-5)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    log_dir = os.path.join(args.log_dir, f"moe_diffusion_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Total params: {total_params:,}")
    print(f"[INFO] Logging to: {log_dir}")
    
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_aux_loss = 0.0
        
        for batch_idx, (obs, action) in enumerate(dataloader):
            obs = obs.to(device)
            action = action.to(device)
            
            batch_size = obs.shape[0]
            t = torch.randint(0, args.timesteps, (batch_size,), device=device)
            
            pred_noise, true_noise, aux_loss = model(action, t, obs)
            
            # 主要损失: 噪声预测MSE
            diffusion_loss = F.mse_loss(pred_noise, true_noise)
            # 总损失: diffusion_loss + lambda * aux_loss
            loss = diffusion_loss + args.aux_loss_weight * aux_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += diffusion_loss.item()
            epoch_aux_loss += aux_loss.item()
            global_step += 1
            
            if global_step % args.log_interval == 0:
                writer.add_scalar("Loss/diffusion", diffusion_loss.item(), global_step)
                writer.add_scalar("Loss/auxiliary", aux_loss.item(), global_step)
                writer.add_scalar("Loss/total", loss.item(), global_step)
                lr_now = optimizer.param_groups[0]['lr']
                print(f"[Step {global_step}] d_loss={diffusion_loss:.6f} aux={aux_loss:.6f} lr={lr_now:.2e}")
        
        scheduler.step()
        epoch_loss /= len(dataloader)
        epoch_aux_loss /= len(dataloader)
        epoch_time = time.time() - epoch_start
        
        # 路由分布分析
        with torch.no_grad():
            sample_obs = obs[:16]
            router_weights = model.analyze_routing(sample_obs)
            # 记录每个Expert的平均权重
            for i in range(args.num_experts):
                writer.add_scalar(f"Router/expert_{i}_weight", 
                                  router_weights[:, i].mean().item(), epoch)
            # 路由熵（越高说明越均匀）
            router_entropy = -(router_weights * torch.log(router_weights + 1e-8)).sum(dim=-1).mean()
            writer.add_scalar("Router/entropy", router_entropy.item(), epoch)
        
        print(f"[Epoch {epoch+1}/{args.epochs}] d_loss={epoch_loss:.6f} aux={epoch_aux_loss:.6f} time={epoch_time:.1f}s")
        
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
    
    final_path = os.path.join(log_dir, "model_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\n[DONE] Final model saved to {final_path}")
    writer.close()


# ============================================================
# 7. 推理演示
# ============================================================
@torch.no_grad()
def demo_sampling(model_path, data_path, device="cuda"):
    model = MoEActionDiffusion(obs_dim=124, action_dim=16, timesteps=100, 
                                num_experts=4, top_k=2, device=device)
    ckpt = torch.load(model_path, map_location=device)
    if "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    
    raw = np.load(data_path)
    obs_data = torch.tensor(raw[:100, :124], dtype=torch.float32, device=device)
    
    # 采样
    gen_actions = model.sample(obs_data)
    
    # 分析路由分布
    router_weights = model.analyze_routing(obs_data)
    print(f"\n[Router Analysis] over {len(obs_data)} samples:")
    for i in range(model.num_experts):
        print(f"  Expert {i}: mean_weight={router_weights[:, i].mean().item():.4f}, "
              f"std={router_weights[:, i].std().item():.4f}")
    
    # 与真实动作对比
    true_actions = raw[1:101, :16] - raw[:100, :16]
    
    print(f"\n[Generation Quality]")
    print(f"  Generated: mean={gen_actions.mean().item():.4f}, std={gen_actions.std().item():.4f}")
    print(f"  Ground Truth: mean={true_actions.mean():.4f}, std={true_actions.std():.4f}")
    print(f"  MSE: {((gen_actions - true_actions)**2).mean().item():.6f}")
    
    return gen_actions.cpu().numpy()


# ============================================================
# 8. Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MoE Action Diffusion Training")
    parser.add_argument("--data_path", type=str, 
                        default=r"D:\projects\data\oakink\processed",
                        help="Path to .npy trajectory file or directory")
    parser.add_argument("--obs_dim", type=int, default=124)
    parser.add_argument("--action_dim", type=int, default=16)
    parser.add_argument("--data_mode", type=str, default="obs_diff")
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=200)
    parser.add_argument("--log_dir", type=str, default=r"D:\projects\data\oakink\processed\logs")
    parser.add_argument("--multi_traj", action="store_true", default=True,
                        help="Enable multi-trajectory mode")
    parser.add_argument("--num_experts", type=int, default=4,
                        help="Number of expert networks")
    parser.add_argument("--top_k", type=int, default=2,
                        help="Number of top experts to activate (sparsity)")
    parser.add_argument("--aux_loss_weight", type=float, default=0.01,
                        help="Weight for load balancing auxiliary loss")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    
    args = parser.parse_args()
    
    if args.demo:
        if args.checkpoint is None:
            print("Please specify --checkpoint for demo mode")
            sys.exit(1)
        demo_sampling(args.checkpoint, args.data_path)
    else:
        train_moe_diffusion(args)
