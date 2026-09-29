import csv
import json
import os
import random

import numpy as np
import torch

import config as cfg


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


def uniform_axis(n_axis, low, high, device, dtype):
    return torch.linspace(float(low), float(high), int(n_axis), device=device, dtype=dtype)


def shifted_axis(n_axis, low, high, shift, device, dtype):
    h = (float(high) - float(low)) / float(n_axis)
    return float(low) + (torch.arange(int(n_axis), device=device, dtype=dtype) + float(shift)) * h


def cartesian_grid_3d_from_axis(axis):
    X, Y, Z = torch.meshgrid(axis, axis, axis, indexing="ij")
    return torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], dim=1)


def cartesian_grid_3d(n_axis, low, high, device, dtype):
    axis = uniform_axis(n_axis, low, high, device, dtype)
    return cartesian_grid_3d_from_axis(axis)


def shifted_grid_3d(n_axis, low, high, shift, device, dtype):
    axis = shifted_axis(n_axis, low, high, shift, device, dtype)
    return cartesian_grid_3d_from_axis(axis)


def midpoint_grid_3d(n_axis, low, high, device, dtype):
    return shifted_grid_3d(n_axis, low, high, 0.5, device, dtype)


def support_axis_1d(n, low, high, device, dtype):
    if cfg.QUADRATURE_RULE == "midpoint":
        h = (float(high) - float(low)) / float(n)
        return float(low) + (torch.arange(int(n), device=device, dtype=dtype) + 0.5) * h
    return torch.linspace(float(low), float(high), int(n), device=device, dtype=dtype)


def quadrature_weights_1d(n, device, dtype):
    n = int(n)
    rule = cfg.QUADRATURE_RULE
    q = torch.ones(n, device=device, dtype=dtype)

    if rule == "midpoint":
        if n < 1:
            raise ValueError("Midpoint requires at least one support point.")
    elif rule == "trapezoid":
        if n < 2:
            raise ValueError("Trapezoid requires at least two support points.")
        q[0] = q[-1] = 0.5
    elif rule == "simpson":
        if n < 3 or (n - 1) % 2 != 0:
            raise ValueError("Simpson requires n = 2P + 1 nodes.")
        q[1:-1:2] = 4.0
        q[2:-1:2] = 2.0
        q /= 3.0
    elif rule == "boole":
        if n < 5 or (n - 1) % 4 != 0:
            raise ValueError("Boole requires n = 4P + 1 nodes.")
        q[0] = q[-1] = 7.0
        q[1:-1:2] = 32.0
        q[2:-1:4] = 12.0
        q[4:-1:4] = 14.0
        q *= 2.0 / 45.0
    else:
        raise ValueError("Unknown quadrature rule.")

    return q


def grid_axes_from_points(x):
    x1 = torch.unique(x[:, 0], sorted=True)
    x2 = torch.unique(x[:, 1], sorted=True)
    x3 = torch.unique(x[:, 2], sorted=True)
    return x1, x2, x3


def mse_and_max_error(pred, target):
    error = pred - target
    mse = torch.mean(error.pow(2)).item()
    max_error = torch.max(torch.abs(error)).item()
    return mse, max_error


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


def sigma_triple_name(sigma_x, sigma_y, sigma_z):
    return "sigmax_%s__sigmay_%s__sigmaz_%s" % (
        float_name(sigma_x),
        float_name(sigma_y),
        float_name(sigma_z),
    )


def normalize_sigmas(sigmas):
    out = {}
    for grade, triple in sigmas.items():
        out[int(grade)] = {
            "sigma_x": float(triple[0]),
            "sigma_y": float(triple[1]),
            "sigma_z": float(triple[2]),
        }
    return out


def sigmas_for_json(sigmas):
    out = {}
    for grade in sorted(sigmas):
        out[str(int(grade))] = {
            "sigma_x": float(sigmas[grade]["sigma_x"]),
            "sigma_y": float(sigmas[grade]["sigma_y"]),
            "sigma_z": float(sigmas[grade]["sigma_z"]),
        }
    return out
