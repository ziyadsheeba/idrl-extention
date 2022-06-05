from abc import ABC, abstractmethod
from typing import Callable, Tuple

import numpy as np
import scipy.optimize

from src.utils import matrix_inverse, multivariate_normal_sample


class ApproximatePosterior(ABC):
    @abstractmethod
    def get_covariance(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def get_mean(self):
        pass


class LaplaceApproximation(ApproximatePosterior):
    def __init__(
        self,
        dim: int,
        prior_covariance: np.ndarray,
        prior_mean: np.ndarray,
        neglog_posterior: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        hessian: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ):
        self.dim = dim
        self.neglog_posterior = neglog_posterior
        self.hessian = hessian
        self._mean = prior_mean
        self._hessian_inv = prior_covariance

    def get_mean(self):
        return self._mean

    def get_covariance(self):
        return self._hessian_inv

    def update(self, X: np.ndarray, y: np.ndarray):
        self._mean, self._hessian_inv = self.simulate_update(X, y)

    def sample(self, n_samples: int = 1) -> np.ndarray:
        return multivariate_normal_sample(
            mu=self.get_mean(), cov=self.get_covariance(), n_samples=n_samples
        )

    def simulate_update(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self._mean is None:
            theta_0 = np.zeros(self.dim)
        else:
            theta_0 = self._mean
        solution = scipy.optimize.minimize(
            self.neglog_posterior, theta_0, args=(y, X), method="L-BFGS-B"
        )
        mean = solution.x
        hess_inv = matrix_inverse(self.hessian(mean, X))
        return np.expand_dims(mean, axis=-1), hess_inv
