import warnings
from abc import ABC, abstractmethod
from typing import Callable, Tuple

import numpy as np
import scipy.optimize
from scipy.sparse import csc_matrix

from src.utils import matrix_inverse, multivariate_normal_sample, timeit


class ApproximatePosterior(ABC):
    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def get_mean(self):
        pass

    @abstractmethod
    def get_covariance(self):
        pass


class LaplaceApproximation(ApproximatePosterior):
    def __init__(
        self,
        prior_covariance: np.ndarray,
        prior_mean: np.ndarray,
        neglog_posterior: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        neglog_posterior_hessian: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ):
        self.neglog_posterior = neglog_posterior
        self.neglog_posterior_hessian = neglog_posterior_hessian
        self._mean = prior_mean
        self._hessian_inv = prior_covariance

    def get_mean(self, project: bool = False, param_norm: float = None):
        if project:
            assert param_norm is not None, "Must provide parameter norm for projection"
            return (
                param_norm * self._mean / np.linalg.norm(self._mean)
                if np.linalg.norm(self._mean) > param_norm
                else self._mean
            )
        else:
            return self._mean

    def get_covariance(self, project: bool = False, param_norm: float = None):
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
            theta_0 = np.zeros(X.shape[0])
        else:
            theta_0 = self._mean
        solution = scipy.optimize.minimize(
            self.neglog_posterior, theta_0, args=(y, X), method="L-BFGS-B", tol=1e-10
        )
        mean = solution.x
        if self.neglog_posterior_hessian is not None:
            hess_inv = matrix_inverse(self.neglog_posterior_hessian(mean, X))
        else:
            hess_inv = solution.hess_inv.todense()
        return np.expand_dims(mean, axis=-1), hess_inv


class GPLaplaceApproximation(ApproximatePosterior):
    def __init__(
        self,
        kernel: Callable,
        prior_mean: Callable,
        neglog_posterior: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        neglog_posterior_hessian: Callable[[np.ndarray, np.ndarray], np.ndarray],
        neglog_posterior_gradient: Callable[
            [np.ndarray, np.ndarray, np.ndarray], np.ndarray
        ],
    ):
        self.kernel = kernel
        self.prior_mean = prior_mean
        self.neglog_posterior = neglog_posterior
        self.neglog_posterior_gradient = neglog_posterior_gradient
        self.neglog_posterior_hessian = neglog_posterior_hessian
        self.f_hat = None

    def get_mean(self, x, X, K_inv: np.ndarray = None):
        if self.f_hat is not None and X is not None:
            if K_inv is None:
                K = self.kernel.eval(X, X)
                K_inv = matrix_inverse(K)
            mean = self.kernel.eval(x, X) @ K_inv @ (self.f_hat - self.prior_mean(X))
        else:
            warnings.warn("Returning Prior Mean")
            mean = self.prior_mean(x)
        return mean

    def get_covariance(
        self,
        x: np.ndarray,
        X: np.ndarray,
        K_inv: np.ndarray = None,
        cov_map: np.ndarray = None,
    ):
        if self.f_hat is not None and X is not None:
            k_x_X = self.kernel.eval(x, X)
            if K_inv is None:
                K = self.kernel.eval(X, X)
                K_inv = matrix_inverse(K)
            if cov_map is None:
                cov_map = matrix_inverse(self.neglog_posterior_hessian(self.f_hat, X))

            k_x_x = self.kernel.eval(x, x)
            cov_map = matrix_inverse(self.neglog_posterior_hessian(self.f_hat, X))
            cov = (
                k_x_x
                - k_x_X @ K_inv @ k_x_X.T
                + k_x_X @ K_inv @ cov_map @ K_inv @ k_x_X.T
            )
        else:
            warnings.warn("Returning Prior Covariance")
            cov = self.kernel.eval(x, x)
        return cov

    def update(
        self,
        X: np.ndarray,
        y: np.ndarray,
        K_inv: np.ndarray = None,
        store: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:

        if K_inv is None:
            K = self.kernel.eval(X, X)
            K_inv = matrix_inverse(K)
        f_x_0 = np.zeros(X.shape[0])
        solution = scipy.optimize.minimize(
            self.neglog_posterior,
            f_x_0,
            args=(X, y, K_inv),
            method="trust-ncg",
            jac=self.neglog_posterior_gradient,
            hess=self.neglog_posterior_hessian,
        ).x
        solution = np.expand_dims(solution, axis=-1)
        if store:
            self.f_hat = solution
        return solution

    def sample(
        self,
        x: np.ndarray,
        X: np.ndarray,
        n_samples: int = 1,
        K_inv: np.ndarray = None,
    ) -> np.ndarray:
        mu = self.get_mean(x, X, K_inv)
        cov = self.get_covariance(x, X, K_inv)
        return multivariate_normal_sample(mu=mu, cov=cov, n_samples=n_samples).squeeze()
