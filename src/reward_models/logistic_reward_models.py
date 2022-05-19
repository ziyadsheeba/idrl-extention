import copy
import json
import multiprocessing
import os
from abc import ABC, abstractmethod
from typing import List, Tuple

import cvxpy as cp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from scipy.integrate import quadrature
from scipy.special import expit

from src.constants import EXPERIMENTS_PATH
from src.constraints.constraints import Constraint
from src.reward_models.approximate_posteriors import (
    ApproximatePosterior,
    LaplaceApproximation,
)
from src.utils import bernoulli_entropy, matrix_inverse, multivariate_normal_sample


class LogisticRewardModel(ABC):
    @abstractmethod
    def neglog_posterior(self):
        pass

    @abstractmethod
    def update_approximate_posterior(self):
        pass

    @abstractmethod
    def neglog_posterior_hessian(self):
        pass

    @abstractmethod
    def get_likelihood(self):
        pass

    @abstractmethod
    def sample_current_approximate_distribution(self):
        pass

    @abstractmethod
    def get_approximate_predictive_distribution(self):
        pass

    @abstractmethod
    def neglog_posterior_bounded_hessian(self):
        pass


class LinearLogisticRewardModel(LogisticRewardModel):
    def __init__(
        self,
        dim: int,
        prior_variance: float,
        param_constraint: Constraint,
        prior_mean: np.ndarray = None,
        approximation: str = "laplace",
        kappa: float = None,
    ):
        """An implementation of a linear reward model.

        Args:
            dim (int): The dimension of the problem.
            prior_variance (float): The weight space prior variance.
            prior_mean (np.ndarray, optional): The weight space prior mean. Defaults to None.
            kappa (float): The hessian lower bound constant.
        """
        self.dim = dim
        if prior_mean is None:
            self.prior_mean = np.zeros(shape=(dim, 1))
        else:
            self.prior_mean = prior_mean

        self.prior_covariance = prior_variance * np.eye(dim)
        self.prior_precision = matrix_inverse(self.prior_covariance)
        self.param_constraint = param_constraint
        if approximation == "laplace":
            self.approximate_posterior = LaplaceApproximation(
                dim=dim,
                prior_covariance=self.prior_covariance,
                prior_mean=self.prior_mean,
                neglog_posterior=self.neglog_posterior,
                hessian=self.neglog_posterior_hessian,
            )
        else:
            raise NotImplementedError(
                f"The approximation '{approximation}' is not implemented."
            )

        self.X = []
        self.y = []
        self.hessian_bound_inv = self.prior_precision
        self.kappa = kappa

    def neglog_posterior(
        self, theta: np.ndarray, y: np.ndarray = None, X: np.ndarray = None
    ) -> float:
        """Returns the negative log posterior excluding the normalization factor.

        Args:
            theta (np.ndarray): The parameter vector.
            y (np.ndarray): The observed binary values.
            X (np.ndarray): The observed covariates.

        Returns:
            float: The negative log posterior excluding the normalization factor.
        """
        if theta.shape == (self.dim,):
            theta = np.expand_dims(theta, axis=-1)

        if y is None and X is None:
            X = np.concatenate(self.X, axis=0)
            y = np.array(self.y)
        elif (y is None and X is not None) or (X is None and y is not None):
            raise ValueError("Specificy X and y, or neither of them.")

        eps = 1e-10
        y_hat = expit(X @ theta).squeeze()
        neg_logprior = (
            0.5
            * (theta - self.prior_mean).T
            @ self.prior_precision
            @ (theta - self.prior_mean)
        ).item()
        neg_loglikelihood = (
            -np.sum(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))
        ).item()
        return neg_logprior + neg_loglikelihood

    def neglog_posterior_hessian(
        self, theta: np.ndarray, X: np.ndarray = None
    ) -> np.ndarray:
        """Returns the hessian of the negative log posterior.

        Args:
            theta (np.ndarray): The parameter vector to compute the hessian at.
            X (np.ndarray, optional): The observed covariates. Defaults to None.

        Returns:
            np.ndarray: The hessian at a specific theta given X.
        """
        if X is None:
            X = np.array(self.X)
        D = np.diag(expit(X @ theta) * (1 - expit(X @ theta)))
        H = X.T @ D @ X + self.prior_covariance
        return H

    def neglog_posterior_hessian_increment(
        self, theta: np.ndarray, x: np.ndarray = None
    ) -> np.ndarray:
        """Returns the hessian of the negative log posterior.

        Args:
            theta (np.ndarray): The parameter vector to compute the hessian at.
            X (np.ndarray, optional): The observed covariates. Defaults to None.

        Returns:
            np.ndarray: The hessian at a specific theta given X.
        """
        X = copy.deepcopy(self.X)
        X.append(x)
        X = np.concatenate(X)
        return self.neglog_posterior_hessian(theta, X)

    def get_likelihood(self, x: np.ndarray, y: int, theta: np.ndarray) -> float:
        """Returns the likelihood of observing (x,y) under the given theta.

        Args:
            x (np.ndarray): The observed covariate
            y (int): The observed binary feedback.
            theta (np.ndarray): The paraemeter value.

        Returns:
            float: The likelihood value.
        """
        y_hat = expit(x @ theta)
        likelihood = y_hat if y == 1 else 1 - y_hat
        return likelihood

    def sample_current_approximate_distribution(
        self, n_samples=1, approximation: str = "laplace"
    ):
        return self.approximate_posterior.sample(n_samples)

    def get_approximate_predictive_distribution(
        self, x: np.ndarray, method="sampling", n_samples: int = None
    ) -> Tuple[float, float]:
        """Computes the approximate predictive distribution.

        Args:
            x (np.ndarray): Input covariate.
            method (str, optional): The method to use, either "sampling" or "quadrature". Defaults to "quadrature".
            n_samples (int, optional): The number of samples to use when method is "sampling". Defaults to None.

        Returns:
            Tuple[float, float]: _description_
        """

        if method == "quadrature":

            mu = x @ self.approximate_posterior.get_mean()
            var = x @ self.approximate_posterior.get_covariance() @ x.T

            def fn(f):
                gaussian = (1 / np.sqrt(2 * np.pi * var)) * np.exp(
                    -0.5 * ((f - mu) ** 2) / var
                )
                return gaussian * expit(f)

            p_1, _ = quadrature(fn, a=-500, b=500, maxiter=1000)
            p_0 = 1 - p_1
        elif method == "sampling":
            if n_samples is None:
                raise ValueError(
                    "Must define the number of samples when using sampling."
                )
            samples = self.sample_current_approximate_distribution(n_samples)
            p_1 = 0
            for i in range(samples.shape[0]):
                sample = samples[i, :]
                p_1 += self.get_likelihood(x=x, y=1, theta=sample)
            p_1 = p_1 / n_samples
            p_0 = 1 - p_1
        else:
            raise NotImplementedError()
        return p_1, p_0

    def neglog_posterior_bounded_hessian(
        self, X: np.ndarray, kappa: float
    ) -> np.ndarray:
        """Returns a bound on the hessian.

        Args:
            X (np.ndarray): The input covariates.
            kappa (float): The scaler replacing the sigmoids second derivatives.

        Returns:
            np.ndarray: The bounded hessian.
        """
        H = X.T @ X * kappa + self.prior_covariance
        return H

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        self.update_approximate_posterior(x, y)
        self.update_inv_hessian_bound(x)

    def update_approximate_posterior(self, x: np.ndarray, y: np.ndarray) -> None:
        """updates the approximate posterior

        Args:
            X (np.ndarray): The input covariates.
            y (np.ndarray): The labels.
        """
        self.X.append(x)
        self.y.append(y)
        X = np.concatenate(self.X, axis=0)
        y = np.array(self.y)
        self.approximate_posterior.update(X, y)

    def update_inv_hessian_bound(self, x: np.ndarray) -> None:
        self.hessian_bound_inv, self.kappa = self.increment_inv_hessian_bound(x)
        print(f"kappa: {self.kappa}")

    def increment_inv_hessian_bound(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        if self.kappa is None:
            kappa = self.compute_hessian_bound(x)
        else:
            kappa_i = self.compute_hessian_bound(x)
            if kappa_i < self.kappa:
                kappa = kappa_i
            else:
                kappa = self.kappa
        X = copy.deepcopy(self.X)
        X.append(x)
        X = np.concatenate(X)
        H_inv = matrix_inverse(self.neglog_posterior_bounded_hessian(X, kappa))
        return H_inv, kappa

    def compute_hessian_bound(self, X: np.ndarray) -> float:
        constraints, theta = self.param_constraint.get_cvxpy_constraint()

        def _get_kappa(x) -> float:
            objective = cp.Maximize(x @ theta)
            problem = cp.Problem(objective, constraints)
            problem.solve()
            kappa = expit(x @ theta.value) * (1 - expit(x @ theta.value))
            return kappa[0][0]

        kappa = np.Inf
        if X.shape == (1, self.dim):
            kappa = _get_kappa(X)
        else:
            for x in X:
                kappa_i = _get_kappa(x)
                if kappa_i < kappa:
                    kappa = kappa_i
        return kappa

    def get_parameters_estimate(self):
        return self.approximate_posterior.get_mean()

    def get_simulated_update(
        self, x: np.ndarray, y: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        X = copy.deepcopy(self.X)
        _y = copy.deepcopy(self.y)
        X.append(x)
        _y.append(y)
        X = np.concatenate(X)
        y = np.array(_y)
        return self.approximate_posterior.simulate_update(X, y)
