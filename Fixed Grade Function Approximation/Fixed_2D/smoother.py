import torch

import config as cfg
from utils import grid_axes_from_points, quadrature_axis_and_weights

class GaussianSmoother2D:

    def __init__(self, x_query, sigma_pair):
        self.x_query = x_query
        self.sigma_x = float(sigma_pair["sigma_x"])
        self.sigma_y = float(sigma_pair["sigma_y"])

        self.x1_query, self.x2_query = grid_axes_from_points(x_query)
        self.support_axis, self.q = quadrature_axis_and_weights(
            cfg.P, cfg.QUADRATURE_RULE,
            cfg.DOMAIN_LOW, cfg.DOMAIN_HIGH,
            device=x_query.device, dtype=x_query.dtype,
        )

        self.Ax = self.weighted_gaussian_matrix(self.x1_query, self.sigma_x)
        self.Ay = self.weighted_gaussian_matrix(self.x2_query, self.sigma_y)
        self.denominator = self.Ax.sum(dim=1).reshape(-1, 1) @ self.Ay.sum(dim=1).reshape(1, -1)

    def weighted_gaussian_matrix(self, x, sigma):
        diff = x.reshape(-1, 1) - self.support_axis.reshape(1, -1)
        kernel = torch.exp(-0.5 * diff.pow(2) / (float(sigma) ** 2))
        return kernel * self.q.reshape(1, -1)

    def apply(self, values_on_support):
        m = self.support_axis.shape[0]
        values = values_on_support.reshape(m, m, -1)
        tiny = torch.finfo(values_on_support.dtype).tiny

        out = []
        for k in range(values.shape[2]):
            numerator = self.Ax @ values[:, :, k] @ self.Ay.t()
            y = numerator / self.denominator.clamp_min(tiny)
            out.append(y.reshape(-1, 1))

        return torch.cat(out, dim=1)
