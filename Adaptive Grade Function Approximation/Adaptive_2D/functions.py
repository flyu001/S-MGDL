import math

import torch


# Adaptive 2D Example 1
def target_hard_2d(x):
    x1 = x[:, 0:1]
    x2 = x[:, 1:2]
    pi = math.pi

    f11 = 0.3 * torch.sin(12*pi*x1 + 4*pi*x1*x1) * torch.abs(
        torch.cos(12*pi*x1 + 14*pi*x1*x1)
    )
    f12 = 0.2 * torch.sin(12*pi*x1 + 12*pi*x1*x2) * torch.abs(
        torch.cos(8*pi*x2 + 12*pi*x1*x1)
    )
    f21 = 0.2 * torch.sin(8*pi*x2 + 6*pi*x1*x2) * torch.abs(
        torch.cos(12*pi*x1 + 8*pi*x2*x2)
    )
    f22 = 0.3 * torch.sin(8*pi*x2 + 10*pi*x2*x2) * torch.abs(
        torch.cos(8*pi*x2 + 10*pi*x2*x2)
    )

    return f11 + f12 + f21 + f22



# Adaptive 2D Example 2
def target_hard2_2d(x):
    x1 = x[:, 0:1]
    x2 = x[:, 1:2]
    pi = math.pi

    f11 = 0.3 * torch.sin(24*pi*x1 + 4*pi*x1*x1) * torch.abs(
        torch.cos(12*pi*x1 + 14*pi*x1*x1)
    )
    f12 = 0.2 * torch.sin(24*pi*x1 + 12*pi*x1*x2) * torch.abs(
        torch.cos(8*pi*x2 + 12*pi*x1*x1)
    )
    f21 = 0.2 * torch.sin(16*pi*x2 + 6*pi*x1*x2) * torch.abs(
        torch.cos(12*pi*x1 + 8*pi*x2*x2)
    )
    f22 = 0.3 * torch.sin(16*pi*x2 + 10*pi*x2*x2) * torch.abs(
        torch.cos(8*pi*x2 + 10*pi*x2*x2)
    )

    return f11 + f12 + f21 + f22
