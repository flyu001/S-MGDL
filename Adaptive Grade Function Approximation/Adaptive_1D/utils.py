import csv
import json
import os
import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def ensure_dir(path):
    if path is None or path == "":
        return
    os.makedirs(path, exist_ok=True)


def sample_uniform_1d(n, low, high, device, dtype):
    return (high - low) * torch.rand(int(n), 1, device=device, dtype=dtype) + low


def data_grid_1d(n, low, high, device, dtype):
    return torch.linspace(float(low), float(high), int(n), device=device, dtype=dtype).reshape(-1, 1)


def midpoint_grid_1d(n, low, high, device, dtype):
    h = (float(high) - float(low)) / float(n)
    x = float(low) + (torch.arange(int(n), device=device, dtype=dtype) + 0.5) * h
    return x.reshape(-1, 1)


def uniform_grid_1d(n, low, high, device, dtype):
    if n < 2:
        raise ValueError("The fixed grid must contain at least two points.")
    return torch.linspace(low, high, n, device=device, dtype=dtype).reshape(-1, 1)


def boole_weights(n, device, dtype):
    if n < 5 or (int(n) - 1) % 4 != 0:
        raise ValueError("Boole's rule requires SUPPORT_GRID_SIZE = 4I + 1.")

    q = torch.empty(int(n), device=device, dtype=dtype)
    q[0] = 7.0
    q[-1] = 7.0

    for j in range(1, int(n) - 1):
        if j % 4 == 0:
            q[j] = 14.0
        elif j % 2 == 0:
            q[j] = 12.0
        else:
            q[j] = 32.0

    return q


def mse_and_max_error(pred, target):
    error = pred - target
    mse = torch.mean(error.pow(2)).item()
    max_error = torch.max(torch.abs(error)).item()
    return mse, max_error


def save_csv(rows, path):
    if rows is None or len(rows) == 0:
        return
    ensure_dir(os.path.dirname(path))
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(payload, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def float_name(x):
    s = "%.4e" % float(x)
    s = s.replace("+0", "")
    s = s.replace("+", "")
    s = s.replace("-0", "-")
    s = s.replace(".", "p")
    return s
