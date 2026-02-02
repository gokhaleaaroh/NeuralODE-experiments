import torch
import torch.nn as nn
from torchdiffeq import odeint
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

class VDP(nn.Module):
    def __init__(self, mu=1.0):
        super().__init__()
        self.mu = mu

    def forward(self, t, state):
        x = state[..., 0]
        y = state[..., 1]
        dx = y
        dy = self.mu * (1.0 - x**2) * y - x

        return torch.stack([dx, dy], dim=1)

class MLPDynamics(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2)
        )

    def forward(self, t, x):
        return self.net(x)


@torch.no_grad()
def generate_vdp_traj(mu=1.0, n_traj=128, t_end=20.0, n_steps=200, init_box=(-3.0, 3.0), noise_std=0.0, method="dopri5", device="cpu"):
    ode = VDP(mu=mu).to(device=device, dtype=torch.float32)

    t = torch.linspace(0.0, t_end, n_steps, device=device, dtype=torch.float32)

    low, high = init_box
    x0 = (high - low) * torch.rand(n_traj, 2, device=device, dtype=torch.float32)

    x = odeint(ode, x0, t, method=method)
    x = x.permute(1, 0, 2).contiguous()

    if noise_std > 0.0:
        x = x + noise_std * torch.randn_like(x)

    return t, x0, x

class TrajectoryData(Dataset):
    def __init__(self, x0, x):
        super().__init__()
        self.x0 = x0
        self.x = x

    def __len__(self):
        return self.x0.shape[0]

    def __getitem__(self, idx):
        return self.x0[idx], self.x[idx]

def save_checkpoint(path, model, optim, step, best_val):
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    torch.save(
        { 
         "model_state": model.state_dict(),
         "optim_state": optim.state_dict(),
         "step": step,
         "best_val": best_val,
        },
        path
    )

def load_checkpoint(path, model, optim=None):
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    if optim is not None and "optim_state" in checkpoint:
        optim.load_state_dict(checkpoint["optim_state"])

    return checkpoint

def rollout(learned_dyn, x0_batch, t):
    pred = odeint(learned_dyn, x0_batch, t, method="dopri5")
    pred = pred.permute(1, 0, 2).contiguous()
    return pred

def train_vdp_neural_ode():
    t_train, x0_train, x_train = generate_vdp_traj(noise_std=0.02)
    t_val, x0_val, x_val = generate_vdp_traj()

    train_loader = DataLoader(
        TrajectoryData(x0_train, x_train),
        batch_size=16,
        shuffle=True,
        drop_last=True,
    )

    learned_dyn = MLPDynamics(hidden=64)
    optim = torch.optim.Adam(learned_dyn.parameters(), lr=1e-3)
    best_val = float("inf")
    best_step = -1

    if os.path.exists("checkpoints/best_vdp_weights.pt"):
        load_checkpoint("checkpoints/best_vdp_weights.pt", learned_dyn, optim=None)
        return learned_dyn, (t_train, x0_train, x_train), (t_val, x0_val, x_val)
    else:
        print("Best checkpoint not found, trainig now")

    learned_dyn.train()
    step = 0
    data_iter = iter(train_loader)
    total_steps = 5000

    while step < total_steps:
        try:
            x0_b, x_true_b = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x0_b, x_true_b = next(data_iter)

        x_pred_b = rollout(learned_dyn, x0_b, t_train)
    
        loss = torch.mean((x_pred_b - x_true_b) ** 2)

        optim.zero_grad(set_to_none=True)
        loss.backward()

        optim.step()

        if (step % 50) == 0:
            learned_dyn.eval()
            with torch.no_grad():
                x_pred_val = rollout(learned_dyn, x0_val, t_val)
                val_loss = torch.mean((x_pred_val - x_val) ** 2).item()

            if val_loss < best_val:
                best_val = val_loss
                best_step = step
                save_checkpoint("checkpoints/best_vdp_weights.pt", learned_dyn, optim, step, best_val)
                print(f"    -> new best! saved to checkpoints/best_vdp_weights.pt")

            learned_dyn.train()
            print(f"step {step:5d} | train loss {loss.item():.6f} | val loss {val_loss:.6f}")

        step += 1

    return learned_dyn, (t_train, x0_train, x_train), (t_val, x0_val, x_val)

@torch.no_grad()
def plot_phase_portrait(t, x0, x_gt, learned_dyn, method="dopri5", max_traj=12, savepath="phase_portrait.png", title="Van der Pol: Ground Truth vs Neural ODE learned dynamics"):
    learned_dyn.eval()
    N = x0.shape[0]
    idx = torch.randperm(N)[: min(max_traj, N)]
    x0_s = x0[idx]
    x_gt_s = x_gt[idx]

    x_pred_s = rollout(learned_dyn, x0_s, t)

    x_gt_np = x_gt_s.detach().numpy()
    x_pr_np = x_pred_s.detach().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
    fig.suptitle(title)

    ax = axes[0]
    for i in range(x_gt_np.shape[0]):
        ax.plot(x_gt_np[i, :, 0], x_gt_np[i, :, 1], linewidth=1.5)
        ax.scatter(x_gt_np[i, 0, 0], x_gt_np[i, 0, 1], s=18)

    ax.set_title("Ground Truth")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for i in range(x_pr_np.shape[0]):
        ax.plot(x_pr_np[i, :, 0], x_pr_np[i, :, 1], linewidth=1.5)
        ax.scatter(x_pr_np[i, 0, 0], x_pr_np[i, 0, 1], s=18)

    ax.set_title("Learned (Neural ODE)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)

    all_x = torch.tensor(
        [x_gt_np[..., 0].min(), x_gt_np[..., 0].max(), x_pr_np[..., 0].min(), x_pr_np[..., 0].max()]
    )
    all_y = torch.tensor(
        [x_gt_np[..., 1].min(), x_gt_np[..., 1].max(), x_pr_np[..., 1].min(), x_pr_np[..., 1].max()]
    )
    xpad = 0.1 * float(all_x.max() - all_x.min() + 1e-6)
    ypad = 0.1 * float(all_y.max() - all_y.min() + 1e-6)
    xlims = (float(all_x.min() - xpad), float(all_x.max() + xpad))
    ylims = (float(all_y.min() - ypad), float(all_y.max() + ypad))
    for ax in axes:
        ax.set_xlim(*xlims)
        ax.set_ylim(*ylims)

    plt.tight_layout()
    plt.savefig(savepath, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved phase portrait to: {savepath}")

@torch.no_grad()
def animate_phase_portrait_gt_vs_learned_gif(
    t: torch.Tensor,
    x0: torch.Tensor,
    x_gt: torch.Tensor,
    f_theta: nn.Module,
    method: str = "dopri5",
    max_traj: int = 12,
    gif_path: str = "phase_gt_vs_pred.gif",
    title: str = "Van der Pol: GT vs Neural ODE (phase space)",
    fps: int = 30,
    stride: int = 1,
    dpi: int = 150,
):

    f_theta.eval()

    device = x0.device
    N, T, _ = x_gt.shape

    # pick subset for readability
    idx = torch.randperm(N, device=device)[: min(max_traj, N)]
    x0_s = x0[idx]         # (B,2)
    x_gt_s = x_gt[idx]     # (B,T,2)

    x_pred_s = rollout(f_theta, x0_s, t)

    # downsample time for faster/smaller GIFs
    frame_idx = torch.arange(0, T, stride, device=device)
    x_gt_s = x_gt_s[:, frame_idx, :]
    x_pred_s = x_pred_s[:, frame_idx, :]
    t_s = t[frame_idx]
    T_s = x_gt_s.shape[1]

    # move to CPU numpy
    gt = x_gt_s.detach().numpy()      # (B,T_s,2)
    pr = x_pred_s.detach().numpy()    # (B,T_s,2)

    # axis limits based on both (pad a bit)
    all_x = np.concatenate([gt[..., 0].ravel(), pr[..., 0].ravel()])
    all_y = np.concatenate([gt[..., 1].ravel(), pr[..., 1].ravel()])
    xpad = 0.08 * (all_x.max() - all_x.min() + 1e-9)
    ypad = 0.08 * (all_y.max() - all_y.min() + 1e-9)
    xlims = (all_x.min() - xpad, all_x.max() + xpad)
    ylims = (all_y.min() - ypad, all_y.max() + ypad)

    fig, ax = plt.subplots(figsize=(5.5, 5.0), dpi=dpi)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(*xlims)
    ax.set_ylim(*ylims)
    ax.grid(True, alpha=0.3)

    # Legend proxies
    ax.plot([], [], color="black", lw=2, label="Ground truth")
    ax.plot([], [], color="crimson", lw=2, linestyle="--", label="Neural ODE")
    ax.legend(loc="best")

    B = gt.shape[0]

    # Create line objects and "current point" markers for each trajectory
    gt_lines = []
    pr_lines = []
    gt_pts = []
    pr_pts = []

    for _ in range(B):
        (l_gt,) = ax.plot([], [], color="black", lw=2)
        (l_pr,) = ax.plot([], [], color="crimson", lw=2, linestyle="--")
        (p_gt,) = ax.plot([], [], marker="o", markersize=4, color="black")
        (p_pr,) = ax.plot([], [], marker="o", markersize=4, color="crimson")
        gt_lines.append(l_gt)
        pr_lines.append(l_pr)
        gt_pts.append(p_gt)
        pr_pts.append(p_pr)

    # A timestamp text (optional but nice)
    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

    def init():
        for i in range(B):
            gt_lines[i].set_data([], [])
            pr_lines[i].set_data([], [])
            gt_pts[i].set_data([], [])
            pr_pts[i].set_data([], [])
        time_text.set_text("")
        return (*gt_lines, *pr_lines, *gt_pts, *pr_pts, time_text)

    def update(frame: int):
        # frame goes 0..T_s-1
        for i in range(B):
            # trajectories up to current frame
            gt_lines[i].set_data(gt[i, : frame + 1, 0], gt[i, : frame + 1, 1])
            pr_lines[i].set_data(pr[i, : frame + 1, 0], pr[i, : frame + 1, 1])

            # current points
            gt_pts[i].set_data([gt[i, frame, 0]], [gt[i, frame, 1]])
            pr_pts[i].set_data([pr[i, frame, 0]], [pr[i, frame, 1]])

        time_text.set_text(f"t = {float(t_s[frame].detach().cpu()):.2f}")
        return (*gt_lines, *pr_lines, *gt_pts, *pr_pts, time_text)

    print("T_s ", T_s)
    anim = FuncAnimation(
        fig,
        update,
        frames=T_s,
        init_func=init,
        interval=1000 / fps,
        blit=True,
    )

    writer = PillowWriter(fps=fps)
    anim.save(gif_path, writer=writer)
    plt.close(fig)
    print(f"Saved GIF to: {gif_path}")

if __name__ == "__main__":
    model, train_data, val_data = train_vdp_neural_ode()
    t_train, x0_train, x_train = train_data
    t_val, x0_val, x_val = val_data

    plot_phase_portrait(t_val, x0_val, x_val, model)
    animate_phase_portrait_gt_vs_learned_gif(
        t=t_val,
        x0=x0_val,
        x_gt=x_val,
        f_theta=model,
        method="dopri5",
        max_traj=10,
        gif_path="phase_gt_vs_pred.gif",
        title=f"Van der Pol Ground Truth vs NeuralODE",
        fps=30,
        stride=1,   # try 2 or 3 if the GIF is too big
        dpi=100,
    )

    print("Done.")

