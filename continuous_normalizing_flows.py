import numpy as np
import math
import torch
import torch.nn as nn
from torchdiffeq import odeint
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os
import imageio.v2 as imageio  # pip install imageio

def sample_spiral(batch_size, noise_std=0.01):
    theta = torch.rand(batch_size) * 2 * 2 * torch.pi
    r = 0.1 + 0.072*theta
    
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    
    samples = torch.stack([x, y], dim=1)
    
    noise = torch.randn_like(samples) * noise_std
    samples = samples + noise
    
    return samples

def logprob_normal(z):
    D = z.shape[1]
    return -0.5 * (z.pow(2).sum(dim=1) + D * math.log(2.0 * math.pi))

class MLPDynamics(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2)
        )

    def forward(self, t, x):
        t_feat = t.expand(x.shape[0], 1)
        return self.net(torch.cat([x, t_feat], dim=1))


def divergence(f, z):
    B, D = z.shape
    div = torch.zeros(B, dtype=torch.float32)
    for i in range(D):
        grad_i = torch.autograd.grad(
            outputs=f[:, i].sum(),
            inputs=z,
            create_graph=True,
            retain_graph=True,
            allow_unused=False,
        )[0]
        div += grad_i[:, i]

    return div.unsqueeze(1)

class CNFAugDynamics(nn.Module):
    def __init__(self, odefunc: nn.Module, divergence_fn=divergence):
        super().__init__()
        self.odefunc = odefunc

    def forward(self, t, y): 
        z = y[:, :2].detach().requires_grad_(True)
        f = self.odefunc(t, z)
        div = divergence(f, z)
        dz_dt = f
        dlogp_dt = -div
        return torch.cat([dz_dt, dlogp_dt], dim=1)


def cnf_logprob_data(x, aug_dynamics, t0=0.0, t1=1.0, method="dopri5", steps=200, rtol=1e-4, atol=1e-4):
    B = x.shape[0]
    y1 = torch.cat([x, torch.zeros(B, 1, dtype=torch.float32)], dim=1)
    # ts = torch.tensor([t1, t0], dtype=torch.float32)
    ts = torch.linspace(t1, t0, steps + 1, dtype=torch.float32)
    ys = odeint(aug_dynamics, y1, ts, method=method, rtol=rtol, atol=atol)
    y0 = ys[-1]                 
    z0 = y0[:, :2]              
    logp_delta_back = y0[:, 2]
    logp0 = logprob_normal(z0)
    return logp0 - logp_delta_back


@torch.no_grad()
def cnf_sample(aug_dynamics, n, t0=0.0, t1=1.0, method="dopri5", steps=40):
    z0 = torch.randn(n, 2)
    y0 = torch.cat([z0, torch.zeros(n, 1)], dim=1)  # (n, 3)

    ts = torch.linspace(t0, t1, steps + 1)
    ys = odeint(aug_dynamics, y0, ts, method=method)
    y1 = ys[-1]
    x = y1[:, :2]
    return x

def cnf_sample_inf(odefunc, n, t0=0.0, t1=1.0, method="dopri5", steps=40):
    z0 = torch.randn(n, 2)
    ts = torch.linspace(t0, t1, steps + 1)
    zs = odeint(odefunc, z0, ts, method=method)
    return zs[-1]

def train_step(aug_dynamics, optimizer, batch_size):
    x = sample_spiral(batch_size)
    logp = cnf_logprob_data(x, aug_dynamics)
    loss = -logp.mean()

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(aug_dynamics.parameters(), 5.0)
    optimizer.step()

    return float(loss.item())

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

@torch.no_grad()
def make_morph_gif(
    odefunc,
    out_path="morph.gif",
    n_points=5000,
    t0=0.0,
    t1=1.0,
    n_frames=48,
    method="dopri5",
    steps=200,              # only used if method == "rk4" (fixed grid)
    fps=20,
    lim=1.3,              # e.g. 3.5 or (xmin, xmax, ymin, ymax). If None, auto from all frames.
    s=2,                   # marker size
    dpi=100,
    device=None,
):
    """
    Creates a morph GIF of samples flowing from base Gaussian to model distribution.

    odefunc: your trained MLPDynamics (or time-conditioned version), signature forward(t, x)->dx/dt
    """

    if device is None:
        device = next(odefunc.parameters()).device

    # --- 1) sample base points
    z0 = torch.randn(n_points, 2, device=device)

    # --- 2) integrate forward and collect frames

    ts = torch.linspace(t0, t1, n_frames, device=device)
    zs = odeint(odefunc, z0, ts, method=method, rtol=1e-4, atol=1e-4)  # (F, N, 2)

    # Choose which time indices become frames
    frames_z = zs
    frames_t = ts

    frames_np = frames_z.detach().cpu().numpy()

    # --- 3) set plot limits (fixed across frames)
    if lim is None:
        # auto from all frames (plus small padding)
        xmin = frames_np[..., 0].min()
        xmax = frames_np[..., 0].max()
        ymin = frames_np[..., 1].min()
        ymax = frames_np[..., 1].max()
        pad_x = 0.05 * (xmax - xmin + 1e-9)
        pad_y = 0.05 * (ymax - ymin + 1e-9)
        xlim = (xmin - pad_x, xmax + pad_x)
        ylim = (ymin - pad_y, ymax + pad_y)
    elif isinstance(lim, (int, float)):
        xlim = (-lim, lim)
        ylim = (-lim, lim)
    else:
        xlim = (lim[0], lim[1])
        ylim = (lim[2], lim[3])

    # --- 4) render frames to images
    tmp_dir = "_morph_frames_tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    frame_files = []

    for k in range(n_frames):
        fig, ax = plt.subplots(figsize=(4, 4), dpi=dpi)
        ax.scatter(frames_np[k, :, 0], frames_np[k, :, 1], s=s)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        # ax.set_xticks([])
        # ax.set_yticks([])
        ax.set_title(f"t = {float(frames_t[k]):.2f}", fontsize=10)

        fname = os.path.join(tmp_dir, f"frame_{k:04d}.png")
        fig.savefig(fname, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        frame_files.append(fname)


    # --- 5) write GIF
    images = [imageio.imread(f) for f in frame_files]
    hold_seconds = 2.0
    frame_duration = 0.1  # seconds per frame
    hold_frames = int(round(hold_seconds / frame_duration))
    frames_with_hold = images + [images[-1]] * hold_frames

    imageio.mimsave(out_path, frames_with_hold, fps=fps, loop=0)

    # --- 6) cleanup temp frames
    for f in frame_files:
        os.remove(f)
    os.rmdir(tmp_dir)

    print(f"Saved GIF to: {out_path}")

if __name__ == "__main__":
    odefunc = MLPDynamics(hidden=128)
    aug = CNFAugDynamics(odefunc)
    opt = torch.optim.Adam(aug.parameters(), lr=5e-4)
    best_loss = float("inf")


    if os.path.exists("checkpoints/best_cnf_weights.pt"):
        load_checkpoint("checkpoints/best_cnf_weights.pt", odefunc, optim=opt)
    else:
        for it in range(5000):
            loss = train_step(aug, opt, batch_size=1024)
            if it % 200 == 0:
                if loss < best_loss:
                    best_loss = loss
                    save_checkpoint("checkpoints/best_cnf_weights.pt", odefunc, opt, it, best_loss)
                print("LOSS: ", it, loss)

    make_morph_gif(
        odefunc,
        out_path="spiral_morph.gif",
        n_points=6000,
        t0=0.0,
        t1=1.00,          # longer horizon often looks cooler
        n_frames=60,
        method="dopri5",
        steps=200,
        fps=20,
        lim=1.0,         # fixed axes for social-friendly visuals
        s=2,
    )
    
    samples = cnf_sample_inf(odefunc, n=5000)  # (5000,2)
    samples = samples.detach().numpy()
    samples_2 = sample_spiral(5000)
    plt.figure(figsize=(8,8))
    plt.scatter(*samples.T, s=10, alpha=1.0, lw=0, c='blue', label="Learned CNF")
    plt.scatter(*samples_2.T, s=10, alpha=1.0, lw=0, c='red', label="Ground Truth")
    plt.legend()
    plt.axis('equal')
    plt.grid(alpha=0.3)
    plt.show()
