import math

import torch


def target_hard_3d(x):
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("target_hard_3d expects x with shape (N,3).")

    x1 = x[:, 0:1]
    x2 = x[:, 1:2]
    x3 = x[:, 2:3]
    variables = [x1, x2, x3]

    pi = math.pi

    a = [
        [0.4, 0.1, 0.5],
        [0.3, 0.4, 0.1],
        [0.3, 0.1, 0.4],
    ]

    b = [
        2.0 * pi,
        8.0 * pi,
        6.0 * pi,
    ]

    c = [
        [2.0 * pi, 1.0 * pi, 3.0 * pi],
        [2.0 * pi, 3.0 * pi, 2.0 * pi],
        [3.0 * pi, 1.0 * pi, 1.0 * pi],
    ]

    d = [
        [3.0 * pi, 4.0 * pi, 1.0 * pi],
        [1.0 * pi, 4.0 * pi, 3.0 * pi],
        [1.0 * pi, 3.0 * pi, 4.0 * pi],
    ]

    value = torch.zeros_like(x1)

    for i in range(3):
        for j in range(3):
            xi = variables[i]
            xj = variables[j]
            value = value + a[i][j] * torch.sin(
                b[i] * xi + c[i][j] * xi * xj
            ) * torch.abs(torch.cos(
                b[j] * xj + d[i][j] * xi.pow(2)
            ))
    return value
