import torch
import torch.nn as nn
from torchdiffeq import odeint
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt

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

    def rollout(x0_batch, t):
        pred = odeint(learned_dyn, x0_batch, t, method="dopri5")
        pred = pred.permute(1, 0, 2).contiguous()
        return pred


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

        x_pred_b = rollout(x0_b, t_train)
    
        loss = torch.mean((x_pred_b - x_true_b) ** 2)

        optim.zero_grad(set_to_none=True)
        loss.backward()

        optim.step()

        if (step % 50) == 0:
            learned_dyn.eval()
            with torch.no_grad():
                x_pred_val = rollout(x0_val, t_val)
                val_loss = torch.mean((x_pred_val - x_val) ** 2).item()

            learned_dyn.train()
            print(f"step {step:5d} | train loss {loss.item():.6f} | val loss {val_loss:.6f}")

        step += 1

    return learned_dyn, (t_train, x0_train, x_train), (t_val, x0_val, x_val)

if __name__ == "__main__":
    model, train_data, val_data = train_vdp_neural_ode()
    print("Done.")
