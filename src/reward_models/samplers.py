import numpy as np
from typing import Callable


class ULA:
    """An implementation of the unadjusted langevin algorithm"""

    def __init__(
        self,
        dim: int,
        posterior: Callable,
        neglog_posterior_gradient: Callable,
        burn_in: int = 1000,
        step_size: float = 0.1,
    ):
        self.dim = dim
        self.posterior = posterior
        self.neglog_posterior_gradient = neglog_posterior_gradient
        self.burn_in = burn_in
        self.step_size = step_size
        self.noise_variance = np.sqrt(2 * step_size)

    def propose(self, x_init: np.array):
        x_current = x_init
        x_current += step_size * self.neglog_posterior_gradient(
            x_current
        ) + self.noise_variance * np.random.randn(dim)
        return x_current

    def sample(self, n_samples: int, x_init: np.array = None):
        if x_init is None:
            x_init = np.zeros(shape=(self.dim,))
        x = x_init
        samples = []
        for _ in range(n_samples):
            for _ in range(self.burn_in):
                x = self.propose(x)
            samples.append(x)
        return samples

if __name__ == "__main__":


    def density_normal(x, mean=0., sigma=1.):
        """
        Density of the normal distribution up to constant
        Args:
            x (n * d): location, can be high dimension
            mean (d): mean
        Returns:
            density value at x
        """
        return np.exp(-np.sum((x - mean)**2, axis=1) / 2 / sigma**2)

    

