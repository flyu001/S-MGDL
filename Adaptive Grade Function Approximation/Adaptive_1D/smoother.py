import torch

from utils import boole_weights

class GaussianSmoother1D:

    def __init__(self, x_query, x_support, sigma):
        if sigma <= 0.0:
            raise ValueError("sigma must be positive.")

        self.x_query = x_query
        self.x_support = x_support
        self.sigma = float(sigma)
        self.weight_matrix = self.make_weight_matrix()

    def make_weight_matrix(self):
        q = boole_weights(
            self.x_support.shape[0],
            device=self.x_query.device,
            dtype=self.x_query.dtype,
        ).reshape(1, -1)

        dist = self.x_query - self.x_support.reshape(1, -1)
        kernel = torch.exp(-0.5 * (dist / self.sigma).pow(2))
        weights = q * kernel
        return weights / torch.sum(weights, dim=1, keepdim=True)

    def apply(self, values_on_support):
        return self.weight_matrix @ values_on_support

