import torch

import config as cfg
from utils import quadrature_weights

class GaussianSmoother1D:

    def __init__(self, x_query, x_support, sigma):
        if sigma <= 0.0:
            raise ValueError("sigma must be positive.")

        self.x_query = x_query
        self.x_support = x_support
        self.sigma = float(sigma)
        self.weight_matrix = self.make_weight_matrix()

    def make_weight_matrix(self):
        q = quadrature_weights(
            cfg.P,
            cfg.QUADRATURE_RULE,
            device=self.x_query.device,
            dtype=self.x_query.dtype,
        ).reshape(1, -1)

        if q.shape[1] != self.x_support.shape[0]:
            raise ValueError("Quadrature nodes and weights do not match.")

        dist = self.x_query - self.x_support.reshape(1, -1)
        kernel = torch.exp(-0.5 * (dist / self.sigma).pow(2))
        weights = q * kernel
        return weights / torch.sum(weights, dim=1, keepdim=True)

    def apply(self, values_on_support):
        return self.weight_matrix @ values_on_support

