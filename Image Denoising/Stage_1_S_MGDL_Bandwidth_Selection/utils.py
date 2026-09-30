import csv
import json
import os

import jax.numpy as jnp
import numpy as onp


def make_validation_masks(height, width, border, block=3):
    val = onp.zeros((height, width), dtype=onp.float32)

    center = block // 2
    rows = onp.arange(border + center, height - border, block)
    cols = onp.arange(border + center, width - border, block)

    val[onp.ix_(rows, cols)] = 1.0
    train = 1.0 - val

    return jnp.asarray(val), jnp.asarray(train)


def mse(pred, target):
    return jnp.mean((pred - target) ** 2)


def masked_mse(pred, target, mask):
    return jnp.sum(mask * (pred - target) ** 2) / jnp.sum(mask)


def max_error(pred, target):
    return float(jnp.max(jnp.abs(pred - target)))


def psnr(pred, target, mask=None):
    if mask is None:
        value = mse(pred, target)
    else:
        value = masked_mse(pred, target, mask)
    return float(-10.0 * jnp.log10(value))


def save_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if len(rows) == 0:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
