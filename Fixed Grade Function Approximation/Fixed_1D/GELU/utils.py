import csv
import json
import os
import random

import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)

def data_grid_1d(n, low, high, device, dtype):
    return torch.linspace(low, high, n, device=device, dtype=dtype).reshape(-1, 1)

def midpoint_grid_1d(n, low, high, device, dtype):
    h = (high - low) / float(n)
    x = low + (torch.arange(n, device=device, dtype=dtype) + 0.5) * h
    return x.reshape(-1, 1)

def uniform_grid_1d(n, low, high, device, dtype):
    if n < 2:
        raise ValueError("The fixed grid must contain at least two points.")
    return torch.linspace(low, high, n, device=device, dtype=dtype).reshape(-1, 1)

def quadrature_grid_1d(p, rule, low, high, device, dtype):
    if p < 1:
        raise ValueError("P must be positive.")

    if rule == "midpoint":
        return midpoint_grid_1d(p, low, high, device, dtype)
    if rule == "trapezoid":
        return uniform_grid_1d(p + 1, low, high, device, dtype)
    if rule == "simpson":
        return uniform_grid_1d(2 * p + 1, low, high, device, dtype)
    if rule == "boole":
        return uniform_grid_1d(4 * p + 1, low, high, device, dtype)
    raise ValueError("Unknown quadrature rule: %s" % rule)


def quadrature_weights(p, rule, device, dtype):
    if p < 1:
        raise ValueError("P must be positive.")

    if rule == "midpoint":
        return torch.ones(p, device=device, dtype=dtype)

    if rule == "trapezoid":
        q = torch.ones(p + 1, device=device, dtype=dtype)
        q[0] = 0.5
        q[-1] = 0.5
        return q

    if rule == "simpson":
        q = torch.ones(2 * p + 1, device=device, dtype=dtype)
        q[1:-1:2] = 4.0
        q[2:-1:2] = 2.0
        return q / 3.0

    if rule == "boole":
        n = 4 * p + 1
        q = torch.empty(n, device=device, dtype=dtype)
        q[0] = 7.0
        q[-1] = 7.0
        for j in range(1, n - 1):
            if j % 4 == 0:
                q[j] = 14.0
            elif j % 2 == 0:
                q[j] = 12.0
            else:
                q[j] = 32.0
        return q

    raise ValueError("Unknown quadrature rule: %s" % rule)


def mse_and_max_error(pred, target):
    err = pred - target
    mse = torch.mean(err.pow(2)).item()
    max_err = torch.max(torch.abs(err)).item()
    return mse, max_err

def save_csv(rows, path):
    if len(rows) == 0:
        return
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def save_json(payload, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def float_name(x):
    s = "%.3e" % float(x)
    s = s.replace("+0", "").replace("+", "")
    s = s.replace("-0", "-")
    s = s.replace(".", "p")
    return s
