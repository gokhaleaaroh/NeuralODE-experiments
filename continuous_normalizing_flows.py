import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def sample_spiral(batch_size, noise_std=0.05):
    theta = torch.rand(batch_size) * 4 * torch.pi
    r = 0.1 + 0.1 * theta
    
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    
    samples = torch.stack([x, y], dim=1)
    
    noise = torch.randn_like(samples) * noise_std
    samples = samples + noise
    
    return samples

samples = sample_spiral(200)
plt.figure(figsize=(8,8))
plt.scatter(*samples.T, s=10, alpha=1.0, lw=0, c='blue')
plt.axis('equal')
plt.grid(alpha=0.3)
plt.show()
