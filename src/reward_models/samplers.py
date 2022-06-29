import warnings
from typing import Callable

import numpy as np


class MALA:
    """An implementation of the metropolis adjusted langevin dynamics algorithm"""

    def __init__(
        self,
        dim: int,
        posterior: Callable,
        neglog_posterior_gradient: Callable,
        burn_in: int = 1000,
        step_size: float = 0.1,
        L: float = None,
    ):
        """_summary_

        Args:
            dim (int): _description_
            posterior (Callable): _description_
            neglog_posterior_gradient (Callable): _description_
            burn_in (int, optional): _description_. Defaults to 1000.
            step_size (float, optional): _description_. Defaults to 0.1.
            L (float, optional): smoothness constant of the negative log posterior gradient. Defaults to None.
        """
        self.dim = dim
        self.posterior = posterior
        self.neglog_posterior_gradient = neglog_posterior_gradient
        self.burn_in = burn_in
        if L is None:
            self.step_size = step_size
        else:
            self.step_size = 0.5 / (L * dim)  # recommended by the paper
            warnings.warn("Overwriting stepsize to use the smoothness bound")

        self.noise_variance = np.sqrt(2 * step_size)

    def propose(self, x_current: np.array) -> np.ndarray:
        """_summary_

        Args:
            x_init (np.array): _description_

        Returns:
            np.ndarray: _description_
        """
        x_proposal = (
            x_current
            - self.step_size * self.neglog_posterior_gradient(x_current)
            + self.noise_variance * np.random.randn(self.dim)
        )
        return x_proposal

    def compute_acceptance_ratio(self, x_current, x_proposal) -> float:
        """_summary_

        Args:
            x_current (_type_): _description_
            x_proposal (_type_): _description_

        Returns:
            float: _description_
        """

        ratio = self.posterior(x_proposal) * MALA.density_normal(
            x=x_current,
            mean=self.propose(x_proposal),
            sigma=self.noise_variance,
        )
        ratio /= self.posterior(x_current) * MALA.density_normal(
            x=x_proposal,
            mean=self.propose(x_current),
            sigma=self.noise_variance,
        )
        return np.minimum(1.0, ratio)

    def sample(self, n_samples: int, x_init: np.ndarray, verbose: bool = True):
        """_summary_

        Args:
            n_samples (int): _description_
            x_init (np.array, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        samples = []
        acceptance_ratio = 0
        x_current = x_init.copy()
        for _ in range(n_samples):
            for _ in range(self.burn_in):
                x_proposal = self.propose(x_current)
                ratio = self.compute_acceptance_ratio(x_current, x_proposal)
                random_uniform_noise = np.random.rand()
                x_current = x_proposal if random_uniform_noise <= ratio else x_current
                if random_uniform_noise <= ratio:
                    acceptance_ratio += 1
            samples.append(x_current)
        if verbose:
            print(f"Acceptance Ratio: {acceptance_ratio/(n_samples*self.burn_in)}")
        return np.vstack(samples)

    @classmethod
    def density_normal(cls, x, mean=0.0, sigma=1.0):
        """
        Density of the normal distribution up to constant
        Args:
            x (n * d): location, can be high dimension
            mean (d): mean
        Returns:
            density value at x
        """
        return np.exp(-np.sum((x - mean) ** 2) / 2 / sigma**2)
