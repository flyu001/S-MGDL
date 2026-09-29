import math

import torch

def target_easy_2d(x):

    x1 = x[:, 0:1]
    x2 = x[:, 1:2]
    pi = math.pi

    f11 = 0.3 * torch.sin(4*pi*x1 + 4*pi*x1*x1) * torch.cos(4*pi*x1 + 8*pi*x1*x1)
    f12 = 0.2 * torch.sin(4*pi*x1 + 8*pi*x1*x2) * torch.cos(8*pi*x2 + 12*pi*x1*x1)
    f21 = 0.2 * torch.sin(8*pi*x2 + 16*pi*x1*x2) * torch.cos(4*pi*x1 + 16*pi*x2*x2)
    f22 = 0.3 * torch.sin(8*pi*x2 + 8*pi*x2*x2) * torch.cos(8*pi*x2 + 12*pi*x2*x2)

    return f11 + f12 + f21 + f22
