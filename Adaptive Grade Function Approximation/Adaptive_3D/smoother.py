import torch

import config as cfg
from utils import boole_weights_1d, grid_axes_from_points_3d, support_axis_1d


class BooleGaussianSmoother3D:

    def __init__(self, x_query, sigma_triple):
        self.x_query = x_query
        self.sigma_x = float(sigma_triple["sigma_x"])
        self.sigma_y = float(sigma_triple["sigma_y"])
        self.sigma_z = float(sigma_triple["sigma_z"])
        self.x1_query, self.x2_query, self.x3_query = grid_axes_from_points_3d(x_query)
        self.support_axis = support_axis_1d(
            cfg.SUPPORT_AXIS_SIZE, cfg.DOMAIN_LOW, cfg.DOMAIN_HIGH,
            device=x_query.device, dtype=x_query.dtype,
        )
        self.q = boole_weights_1d(cfg.SUPPORT_AXIS_SIZE, x_query.device, x_query.dtype)
        self.Ax = self.weighted_gaussian_matrix(self.x1_query, self.sigma_x)
        self.Ay = self.weighted_gaussian_matrix(self.x2_query, self.sigma_y)
        self.Az = self.weighted_gaussian_matrix(self.x3_query, self.sigma_z)
        self.dx = self.Ax.sum(dim=1)
        self.dy = self.Ay.sum(dim=1)
        self.dz = self.Az.sum(dim=1)

    def weighted_gaussian_matrix(self, x, sigma):
        diff = x.reshape(-1, 1) - self.support_axis.reshape(1, -1)
        kernel = torch.exp(-0.5 * diff.pow(2) / (sigma ** 2))
        return kernel * self.q.reshape(1, -1)

    def apply(self, values_on_support):
        m = int(cfg.SUPPORT_AXIS_SIZE)
        values = values_on_support.reshape(m, m, m, -1)
        tiny = torch.finfo(values_on_support.dtype).tiny
        denominator = (
            self.dx.reshape(-1, 1, 1)
            * self.dy.reshape(1, -1, 1)
            * self.dz.reshape(1, 1, -1)
        ).clamp_min(tiny)
        out = []
        for k in range(values.shape[3]):
            # Contract one coordinate at a time; no dense 3D kernel is formed.
            temp_x = torch.einsum("ai,ijk->ajk", self.Ax, values[:, :, :, k])
            temp_y = torch.einsum("bj,ajk->abk", self.Ay, temp_x)
            numerator = torch.einsum("ck,abk->abc", self.Az, temp_y)
            out.append((numerator / denominator).reshape(-1, 1))
        return torch.cat(out, dim=1)

def grade_output_on_support_grid(model, grade, support_axis):
    m = int(support_axis.shape[0])

    X, Y = torch.meshgrid(support_axis, support_axis, indexing="ij")
    xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)

    values = torch.empty(
        m,
        m,
        m,
        device=support_axis.device,
        dtype=support_axis.dtype,
    )

    for k in range(m):
        z = support_axis[k].expand(xy.shape[0], 1)
        points = torch.cat([xy, z], dim=1)
        values[:, :, k] = model.u_raw(points, grade).reshape(m, m)

    return values
