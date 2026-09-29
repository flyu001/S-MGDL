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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)

def cartesian_grid_2d(n, low, high, device, dtype):
    t = torch.linspace(float(low), float(high), int(n), device=device, dtype=dtype)
    X, Y = torch.meshgrid(t, t, indexing="ij")
    return torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)

def shifted_grid_2d(n, low, high, shift, device, dtype):
    h = (float(high) - float(low)) / float(n)
    t = float(low) + (torch.arange(int(n), device=device, dtype=dtype) + float(shift)) * h
    X, Y = torch.meshgrid(t, t, indexing="ij")
    return torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)

def midpoint_grid_2d(n, low, high, device, dtype):
    return shifted_grid_2d(n, low, high, 0.5, device, dtype)

def quadrature_axis_and_weights(p, rule, low, high, device, dtype):
    p = int(p)
    if p < 1:
        raise ValueError("P must be a positive integer.")

    if rule == "midpoint":
        h = (float(high) - float(low)) / p
        z = float(low) + (torch.arange(p, device=device, dtype=dtype) + 0.5) * h
        q = torch.ones(p, device=device, dtype=dtype)

    elif rule == "trapezoid":
        z = torch.linspace(float(low), float(high), p + 1, device=device, dtype=dtype)
        q = torch.ones(p + 1, device=device, dtype=dtype)
        q[0] = 0.5
        q[-1] = 0.5

    elif rule == "simpson":
        m = 2 * p + 1
        z = torch.linspace(float(low), float(high), m, device=device, dtype=dtype)
        q = torch.ones(m, device=device, dtype=dtype)
        q[1:-1:2] = 4.0
        q[2:-1:2] = 2.0
        q = q / 3.0

    elif rule == "boole":
        m = 4 * p + 1
        z = torch.linspace(float(low), float(high), m, device=device, dtype=dtype)
        q = torch.zeros(m, device=device, dtype=dtype)
        q[0] = 7.0
        q[-1] = 7.0
        for i in range(1, m - 1):
            r = i % 4
            if r == 1 or r == 3:
                q[i] = 32.0
            elif r == 2:
                q[i] = 12.0
            else:
                q[i] = 14.0
        q = (2.0 / 45.0) * q

    else:
        raise ValueError("QUADRATURE_RULE must be midpoint, trapezoid, simpson, or boole.")

    # The common mesh-width factor cancels in the normalized smoother.
    return z, q


def support_grid_2d(p, rule, low, high, device, dtype):
    z, _ = quadrature_axis_and_weights(p, rule, low, high, device, dtype)
    X, Y = torch.meshgrid(z, z, indexing="ij")
    return torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)

def grid_axes_from_points(x):
    x1 = torch.unique(x[:, 0], sorted=True)
    x2 = torch.unique(x[:, 1], sorted=True)
    return x1, x2

def mse_and_max_error(pred, target):
    err = pred - target
    return torch.mean(err.pow(2)).item(), torch.max(torch.abs(err)).item()

def save_csv(rows, path):
    if len(rows) == 0:
        return
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def save_json(data, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def float_name(x):
    s = "%.3e" % float(x)
    s = s.replace("+0", "").replace("+", "")
    s = s.replace("-0", "-")
    s = s.replace(".", "p")
    return s

def sigma_pair_name(sigma_x, sigma_y):
    return "sigmax_%s__sigmay_%s" % (float_name(sigma_x), float_name(sigma_y))

def normalize_sigmas(sigmas):
    out = {}
    for grade, pair in sigmas.items():
        out[int(grade)] = {
            "sigma_x": float(pair[0]),
            "sigma_y": float(pair[1]),
        }
    return out

def sigmas_for_json(sigmas):
    out = {}
    for grade in sorted(sigmas):
        out[str(int(grade))] = {
            "sigma_x": float(sigmas[grade]["sigma_x"]),
            "sigma_y": float(sigmas[grade]["sigma_y"]),
        }
    return out
