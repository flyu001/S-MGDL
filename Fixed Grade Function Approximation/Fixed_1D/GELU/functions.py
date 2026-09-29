import math
import torch

def target_easy_1d(x):
    if x.ndim != 2 or x.shape[1] != 1:
        raise ValueError("target_easy_1d expects x with shape (N,1).")
    return torch.sin(32.0 * math.pi * x) - 0.5 * torch.cos(16.0 * math.pi * x.pow(2))
