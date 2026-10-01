import math

import torch


# Adaptive 1D Example 1
def target_hard_1d(x):
    oscillatory = 0.6 * torch.sin(200.0 * math.pi * x) + 0.8 * torch.cos(160.0 * math.pi * x.pow(2))
    envelope = (1.0 + 8.0 * x.pow(8)) / (1.0 + 10.0 * x.pow(4))
    saw = torch.abs(180.0 * x - 2.0 * torch.floor((180.0 * x + 1.0) / 2.0))
    return oscillatory + envelope * saw


# Adaptive 1D Example 2
def target_hard2_1d(x):
    oscillatory_1 = 0.55 * torch.sin(170.0 * math.pi * x + 0.4)
    oscillatory_2 = 0.75 * torch.cos(130.0 * math.pi * x.pow(2) - 0.2)
    envelope = 0.35 + 0.65 * torch.exp(-7.0 * x.pow(2))
    nonsmooth = torch.abs(130.0 * x - 2.0 * torch.floor((130.0 * x + 1.0) / 2.0))
    return oscillatory_1 + oscillatory_2 + envelope * nonsmooth
