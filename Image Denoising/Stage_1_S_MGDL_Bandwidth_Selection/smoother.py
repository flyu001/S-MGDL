import jax.numpy as jnp
import numpy as np

import config as cfg


def boole_weights(n, dtype=jnp.float32):
    if (n - 1) % 4 != 0:
        raise ValueError("SUPPORT_AXIS_SIZE must be 4I+1.")

    w = np.zeros(n, dtype=np.float64)
    w[0] = 7.0
    w[-1] = 7.0

    for i in range(1, n - 1):
        r = i % 4
        if r == 1 or r == 3:
            w[i] = 32.0
        elif r == 2:
            w[i] = 12.0
        else:
            w[i] = 14.0

    return jnp.asarray((2.0 / 45.0) * w, dtype=dtype)


def support_grid(dtype=jnp.float32):
    axis = jnp.linspace(
        cfg.DOMAIN_LOW,
        cfg.DOMAIN_HIGH,
        cfg.SUPPORT_AXIS_SIZE,
        dtype=dtype,
    )
    x, y = jnp.meshgrid(axis, axis, indexing="xy")
    return axis, jnp.stack([x, y], axis=-1)


def smooth_2d(values, query_x, query_y, sigma_x, sigma_y, support_axis):
    q = boole_weights(len(support_axis), dtype=values.dtype)

    dx = (query_x[:, None] - support_axis[None, :]) / sigma_x
    dy = (query_y[:, None] - support_axis[None, :]) / sigma_y

    Ax = q[None, :] * jnp.exp(-0.5 * dx ** 2)
    Ay = q[None, :] * jnp.exp(-0.5 * dy ** 2)

    numerator = Ay @ values @ Ax.T
    denominator = jnp.outer(jnp.sum(Ay, axis=1), jnp.sum(Ax, axis=1))

    return numerator / denominator
