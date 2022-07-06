import copy
from abc import ABC, abstractmethod
from typing import Callable, List, Tuple, Union

import cvxpy as cp
import numpy as np
from scipy.integrate import quadrature
from scipy.linalg import block_diag
from scipy.special import expit

from src.constraints.constraints import Constraint
from src.reward_models.approximate_posteriors import (
    ApproximatePosterior,
    GPLaplaceApproximation,
    LaplaceApproximation,
)
from src.reward_models.kernels import Kernel, LinearKernel, RBFKernel
from src.reward_models.samplers import MALA
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
    def likelihood(self):
        pass

    @abstractmethod
    def sample_current_approximate_distribution(self):
        pass

    # @abstractmethod
    # def get_approximate_predictive_distribution(self):
    #     pass

    # @abstractmethod
    # def neglog_posterior_bounded_hessian(self):
    #     pass


class LinearLogisticRewardModel(LogisticRewardModel):
    def __init__(
        self,
        dim: int,
        prior_variance: float,
        x_min: Union[List[float], float],
        x_max: Union[List[float], float],
        param_norm: float = 1,
        prior_mean: np.ndarray = None,
        approximation: str = "laplace",
        mcmc_sampler: str = "mala",
    ):
        """_summary_

        Args:
            dim (int): Dimensionality.
            prior_variance (float): Parameter's prior variance.
            x_min (Union[List[float], float]): The minimum state (in terms of the norm), full list or coordinate wise bound.
            x_max (Union[List[float], float]): The maximum state (in terms of the norm), full list or coordinate wise bound.
            param_norm (float, optional): The parameter norm bound. Defaults to 1.
            prior_mean (np.ndarray, optional): The prior mean for the parameter vector. Defaults to None.
            approximation (str, optional): The approximation algorithm. Defaults to "laplace".
            sampler(str, optional): The mcmc sampler to be used. defaults to "mala"".

        Raises:
            NotImplementedError: If approximate posterior is not implemented.
        """

        self.dim = dim
        if prior_mean is None:
            self.prior_mean = np.zeros(shape=(dim, 1))
        else:
            self.prior_mean = prior_mean

        self.prior_covariance = prior_variance * np.eye(dim)
        self.prior_precision = matrix_inverse(self.prior_covariance)
        self.X = []
        self.y = []
        self.kappas = []
        self.hessian_bound_inv = self.prior_covariance
        self.hessian_bound_coord_inv = self.prior_covariance
        self.param_norm = param_norm

        if isinstance(x_min, float):
            x_min = dim * [x_min]
        if isinstance(x_max, float):
            x_max = dim * [x_max]
        self.x_min = np.expand_dims(np.array(x_min), axis=0)
        self.x_max = np.expand_dims(np.array(x_max), axis=0)
        self.kappa = self.compute_uniform_kappa()

        if approximation == "laplace":
            self.approximate_posterior = LaplaceApproximation(
                prior_covariance=self.prior_covariance,
                prior_mean=self.prior_mean,
                neglog_posterior=self.neglog_posterior,
                neglog_posterior_hessian=self.neglog_posterior_hessian,
            )
        else:
            raise NotImplementedError(
                f"The approximation '{approximation}' is not implemented."
            )
        if mcmc_sampler == "mala":
            self.sampler = MALA(
                dim=dim,
                posterior=self.posterior,
                neglog_posterior_gradient=self.neglog_posterior_gradient,
            )

    def neglog_posterior(
        self, theta: np.ndarray, y: np.ndarray = None, X: np.ndarray = None
    ) -> float:
        """Returns the negative log posterior excluding up to normalization constant.

        Args:
            theta (np.ndarray): The parameter vector.
            y (np.ndarray): The observed binary values.
            X (np.ndarray): The observed covariates.

        Returns:
            float: The negative log posterior excluding the normalization factor.
        """
        if theta.shape == (self.dim,):
            theta = np.expand_dims(theta, axis=-1)
        neg_logprior = (
            0.5
            * (theta - self.prior_mean).T
            @ self.prior_precision
            @ (theta - self.prior_mean)
        ).item()
        if y is None and X is None:
            if len(self.X) > 0:
                X = np.concatenate(self.X, axis=0)
                y = np.array(self.y)
            else:
                assert len(self.y) == 0, "No covariates by labels exist in memory"
                return neg_logprior
        elif (y is None and X is not None) or (X is None and y is not None):
            raise ValueError("Specificy X and y, or neither of them.")

        eps = 1e-10
        y_hat = expit(X @ theta).squeeze()
        neg_loglikelihood = (
            -np.sum(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))
        ).item()
        return neg_logprior + neg_loglikelihood

    def neglog_posterior_gradient(
        self, theta: np.ndarray, y: np.ndarray = None, X: np.ndarray = None
    ):
        if theta.shape == (self.dim,):
            theta = np.expand_dims(theta, axis=-1)
        grad_neglog_prior = self.prior_precision @ (theta - self.prior_mean)
        if y is None and X is None:
            if len(self.X) > 0:
                X = np.concatenate(self.X, axis=0)
                y = np.array(self.y)
            else:
                assert len(self.y) == 0, "No covariates by labels exist in memory"
                return grad_neglog_prior.squeeze()

        elif (y is None and X is not None) or (X is None and y is not None):
            raise ValueError("Specificy X and y, or neither of them.")

        y_hat = expit(X @ theta).squeeze()
        grad_neglog_likelihood = -np.sum(X * np.expand_dims(y - y_hat, axis=1))
        return grad_neglog_prior.squeeze() + grad_neglog_likelihood.squeeze()

    def posterior(
        self, theta: np.ndarray, y: np.ndarray = None, X: np.ndarray = None
    ) -> float:
        return np.exp(-self.neglog_posterior(theta=theta, y=y, X=X))

    def likelihood(self, x: np.ndarray, y: int, theta: np.ndarray) -> float:
        """Returns the likelihood of observing (x,y) under the given theta.

        Args:
            x (np.ndarray): The observed covariate
            y (int): The observed binary feedback.
            theta (np.ndarray): The paraemeter value.

        Returns:
            float: The likelihood value.
        """
        y_hat = expit(x @ theta).item()
        likelihood = y_hat if y == 1 else 1 - y_hat
        return likelihood

    def increment_neglog_posterior(
        self,
        theta: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """Returns the negative log posterior incremented by x excluding the normalization factor.

        Args:
            theta (np.ndarray): The parameter vector.
            x (np.ndarray): The covariate to add to the dataset.
            y (np.ndarray): The observed binary value.

        Returns:
            float: The negative log posterior excluding the normalization factor.
        """
        X = copy.deepcopy(self.X)
        X.append(x)
        _y = copy.deepcopy(self.y)
        _y.append(y)

        X = np.concatenate(X)
        y = np.array(_y)
        if theta.shape == (self.dim,):
            theta = np.expand_dims(theta, axis=-1)
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
            if len(self.X) > 0:
                X = np.concatenate(self.X)
            else:
                return self.prior_precision
        D = np.eye(X.shape[0]) * (expit(X @ theta) * (1 - expit(X @ theta)))
        D = np.atleast_2d(D)
        H = X.T @ D @ X + self.prior_precision
        return H

    def neglog_posterior_bounded_hessian(self, X: np.ndarray) -> np.ndarray:
        """Returns a bound on the hessian.

        Args:
            X (np.ndarray): The input covariates.
            kappa (float): The scaler replacing the sigmoids second derivatives.

        Returns:
            np.ndarray: The bounded hessian.
        """
        H = X.T @ X * self.kappa + self.prior_precision
        return H

    def neglog_posterior_bounded_coordinate_hessian(
        self, X: np.ndarray, kappas: list
    ) -> np.ndarray:
        """Returns the coordinate bounded hessian.

        Args:
            X (np.ndarray): Covariates.
            kappas (list): Variance bounds.

        Returns:
            np.ndarray: The coordinate-bounded hessian.
        """
        H = X.T @ np.diag(kappas) @ X + self.prior_precision
        return H

    def increment_neglog_posterior_hessian(self, theta, x: np.ndarray) -> np.ndarray:
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
        D = np.eye(X.shape[0]) * (expit(X @ theta) * (1 - expit(X @ theta)))
        D = np.atleast_2d(D)
        H = X.T @ D @ X + self.prior_precision
        return H

    def increment_neglog_posterior_hessian_bound(
        self, x: np.ndarray, kappa: float
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
        H = kappa * X.T @ X + self.prior_precision
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

    def sample_current_approximate_distribution(self, n_samples=1):
        return self.approximate_posterior.sample(n_samples)

    def sample_mcmc(self, n_samples) -> np.ndarray:
        """Samples from the given mcmc sampler. Initializes the sampler by the posterior mode.

        Args:
            n_samples (_type_): number of samples

        Returns:
            np.ndarray: Samples from the true posterior.
        """
        return self.sampler.sample(
            n_samples=n_samples,
            x_init=copy.deepcopy(self.get_parameters_estimate().squeeze()),
        )

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
                p_1 += self.likelihood(x=x, y=1, theta=sample)
            p_1 = p_1 / n_samples
            p_0 = 1 - p_1
        else:
            raise NotImplementedError()
        return p_1, p_0

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        """Updates the reward model after a new observation (x,y)

        Args:
            x (np.ndarray): The observed covariate.
            y (np.ndarray): The observed response.
        """
        self.update_inv_hessian_bound(x)
        self.update_inv_hessian_coordinate_bound(x)
        self.update_approximate_posterior(x, y)

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
        self.hessian_bound_inv = self.increment_inv_hessian_bound(x)

    def update_inv_hessian_coordinate_bound(self, x: np.ndarray) -> None:
        (
            self.hessian_bound_coord_inv,
            self.kappas,
        ) = self.increment_inv_hessian_coordinate_bound(x)

    def increment_inv_hessian_bound(self, x: np.ndarray) -> np.ndarray:
        X = copy.deepcopy(self.X)
        X.append(x)
        X = np.concatenate(X)
        H_inv = matrix_inverse(self.neglog_posterior_bounded_hessian(X))
        return H_inv

    def increment_inv_hessian_coordinate_bound(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        kappa_i = self.compute_uniform_kappa(x)
        kappas = copy.deepcopy(self.kappas)
        kappas.append(kappa_i)
        X = copy.deepcopy(self.X)
        X.append(x)
        X = np.concatenate(X)
        H_inv = matrix_inverse(
            self.neglog_posterior_bounded_coordinate_hessian(X, kappas)
        )
        return H_inv, kappas

    def compute_uniform_kappa(self, x: np.ndarray = None) -> float:
        """Computes a uniform lowerbound on the variances using the parameter bound.

        Args:
            x (np.ndarray, optional): The covariate. Defaults to None.

        Returns:
            float: The variance bound.
        """
        if x is None:
            if np.linalg.norm(self.x_min) >= np.linalg.norm(self.x_max):
                x = self.x_min
            else:
                x = self.x_max
        theta = (
            self.param_norm * x.T / np.linalg.norm(x)
            if np.linalg.norm(x) > 1e-9
            else x.T
        )
        kappa = expit(x @ theta) * (1 - expit(x @ theta))
        return kappa.item()

    def compute_set_kappa(
        self, X: np.ndarray, constraint: Constraint, return_kappa_list: bool = False
    ) -> Union[float, List]:
        """Computes the variance bound given a parameter set.

        Args:
            X (np.ndarray): The covariates.
            constraint (Constraint): The parameter set expressed as a constraint.
            return_kappa_list (bool, optional): Whether to return the variance list for each covariate. Defaults to False.

        Returns:
            Union[float, List]: uniform bound or bound per covariate.
        """
        constraints, theta = constraint.get_cvxpy_constraint()
        _x = cp.Parameter((1, self.dim))
        objective = cp.Maximize(_x @ theta)
        problem = cp.Problem(objective, constraints)

        def _get_kappa(x) -> float:
            _x.value = x
            problem.solve()
            kappa = expit(x @ theta.value) * (1 - expit(x @ theta.value))
            return kappa.item()

        kappa = np.Inf
        kappas = []
        if X.shape == (1, self.dim):
            kappa = _get_kappa(X)
            kappas.append(kappa)
        else:
            for i in range(len(X)):
                x = np.expand_dims(X[i, :], axis=0)
                kappa_i = _get_kappa(x)
                if return_kappa_list:
                    kappas.append(kappa_i)
                if kappa_i < kappa:
                    kappa = kappa_i
        if return_kappa_list:
            return kappas
        else:
            return kappa

    @classmethod
    def from_trajectories_to_pseudo_states(cls, trajectories: np.ndarray) -> np.ndarray:
        """Transforms trajectories to effective states.

        Args:
            trajectories (np.ndarray): A 3d array, (row, columns) is a trajectory.

        Returns:
            np.ndarray: A 2d array of the effective states, (n_states, dim)
        """
        states = np.sum(trajectories, axis=0).T
        return states

    def get_parameters_estimate(self, project: bool = False):
        return self.approximate_posterior.get_mean(
            project=project, param_norm=self.param_norm
        )

    def get_parameters_covariance(self):
        return self.approximate_posterior.get_covariance()

    def get_parameters_moments(self):
        return self.get_parameters_estimate(), self.get_parameters_covariance()

    def get_dataset(self):
        return copy.deepcopy(self.X), copy.deepcopy(self.y)

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


class GPLogisticRewardModel(LogisticRewardModel):
    def __init__(
        self,
        dim: int,
        kernel: Kernel,
        reward_min: float = -1,
        reward_max: float = 1,
        prior_mean: Callable = None,
        approximation: str = "laplace",
    ):
        """_summary_

        Args:
            dim (int): Dimensionality of the covariates.
            kernel (Kernel): A the similarity kernel.
            reward_min (float): The minimum reward possible.
            reward_max (float): The maximum reward possible.
            prior_mean (np.ndarray, optional): The prior mean for the parameter vector. Defaults to None.
            approximation (str, optional): The approximation algorithm. Defaults to "laplace".

        Raises:
            NotImplementedError: If approximate posterior is not implemented.
        """

        self.dim = dim
        self.kernel = kernel
        if prior_mean is None:
            self.prior_mean = lambda x: np.expand_dims(
                np.array(x.shape[0] * [0]), axis=-1
            )

        if approximation == "laplace":
            self.approximate_posterior = GPLaplaceApproximation(
                kernel=self.kernel,
                prior_mean=self.prior_mean,
                neglog_posterior=self.neglog_posterior,
                neglog_posterior_hessian=self.neglog_posterior_hessian,
                neglog_posterior_gradient=self.neglog_posterior_gradient,
            )
        else:
            raise NotImplementedError(
                f"Approximation method {approximation} not implemented"
            )
        self.X = []
        self.y = []
        self.K_inv = None
        self.cov_map = None

    def likelihood(self, f_x: np.ndarray, y: np.ndarray):
        """Likelihood function. Assumes an explicit ordering of f_x, that each 2 consecutive rows correspond to the same query.

        Args:
            f_x (np.ndarray): _description_
            y (np.ndarray): _description_

        Returns:
            _type_: _description_
        """
        f_x_reduced = np.array([f_x[i] - f_x[i + 1] for i in range(0, len(f_x), 2)])
        y_hat = expit(f_x_reduced)
        return -np.prod(y_hat**y + (1 - y_hat) ** (1 - y))

    def neglog_likelihood(self, f_x: np.ndarray, y: np.ndarray):
        """Likelihood function. Assumes an explicit ordering of f_x, that each 2 consecutive rows correspond to the same query.

        Args:
            f_x (np.ndarray): _description_
            y (np.ndarray): _description_

        Returns:
            _type_: _description_
        """
        f_x_reduced = np.array(
            [f_x[i] - f_x[i + 1] for i in range(0, len(f_x), 2)]
        ).squeeze()
        eps = 1e-10
        y_hat = expit(f_x_reduced)
        return -np.sum(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))

    def neglog_posterior(
        self,
        f_x: np.ndarray,
        X: np.ndarray = None,
        y: np.ndarray = None,
        K_inv: np.ndarray = None,
    ):
        """_summary_

        Args:
            f_x (np.ndarray): _description_
            y (np.ndarray, optional): _description_. Defaults to None.
            X (np.ndarray, optional): _description_. Defaults to None.
            K_inv (np.ndarray, optional): Inverse Gram Matrix. Using this argument significantly
                increases update speed. Defaults to None.

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        if y is None and X is None:
            if len(self.X) > 0:
                X = np.concatenate(self.X, axis=0)
                y = np.array(self.y)
            else:
                raise ValueError(
                    "The memory is empty, must pass explicit covariates and labels."
                )
        elif (y is None and X is not None) or (X is None and y is not None):
            raise ValueError("Specificy X and y, or neither of them.")
        if f_x.ndim == 1:
            f_x = np.expand_dims(f_x, axis=-1)
        prior_mean = self.prior_mean(X)

        if K_inv is None:
            K = self.kernel.eval(X, X)
            K_inv = matrix_inverse(K)
        neglog_prior = (0.5 * (f_x - prior_mean).T @ K_inv @ (f_x - prior_mean)).item()
        neglog_likelihood = self.neglog_likelihood(f_x, y)
        return neglog_prior + neglog_likelihood

    def neglog_posterior_gradient(
        self,
        f_x: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        K_inv: np.ndarray,
    ) -> np.ndarray:
        if f_x.ndim == 1:
            f_x = np.expand_dims(f_x, axis=-1)
        prior_mean = self.prior_mean(X)
        grad_prior = K_inv @ (f_x - prior_mean)
        f_x_reduced = np.array(
            [f_x[i] - f_x[i + 1] for i in range(0, len(f_x), 2)]
        ).squeeze()
        y_hat = expit(f_x_reduced)
        grad_likelihood = y - y_hat
        grad_likelihood = np.repeat(grad_likelihood, repeats=2, axis=0).squeeze()
        grad_likelihood[::2] *= -1
        grad = grad_likelihood.squeeze() + grad_prior.squeeze()
        return grad

    def neglog_posterior_hessian(
        self,
        f_x: np.ndarray,
        X: np.ndarray = None,
        y: np.ndarray = None,
        K_inv: np.ndarray = None,
    ):
        if X is None:
            if len(self.X) > 0:
                X = np.concatenate(self.X, axis=0)
            else:
                raise ValueError(
                    "The memory is empty, must pass explicit covariates and labels."
                )
        if K_inv is None:
            K = self.kernel.eval(X, X)
            K_inv = matrix_inverse(K)
        W = self.neglog_likelihood_hessian(f_x)
        return W + K_inv

    def neglog_likelihood_hessian(self, f_x: np.ndarray):
        f_x_reduced = np.array([f_x[i] - f_x[i + 1] for i in range(0, len(f_x), 2)])
        W = []
        for f_delta in f_x_reduced:
            variance = (expit(f_delta) * (1 - expit(f_delta))).item()
            W.append(np.array([[variance, -variance], [-variance, variance]]))
        W = block_diag(*W)
        return W

    def update(self, x_1: np.ndarray, x_2, y: np.ndarray) -> None:

        """Updates the reward model after a new observation (x,y)

        Args:
            x (np.ndarray): The observed covariate.
            y (np.ndarray): The observed response.
        """
        if x_1.ndim == 1:
            x_1 = np.expand_dims(x_1, axis=0)
        if x_2.ndim == 1:
            x_2 = np.expand_dims(x_2, axis=0)
        self.X.append(x_1)
        self.X.append(x_2)
        self.y.append(y)
        f_x = self.update_approximate_posterior()

        self.update_gram_matrix_inverse()
        self.update_map_covariance(f_x)

    def update_approximate_posterior(self) -> np.ndarray:
        """updates the approximate posterior

        Args:
            X (np.ndarray): The input covariates.
            y (np.ndarray): The labels.
        """
        X = np.concatenate(self.X, axis=0)
        y = np.array(self.y)
        return self.approximate_posterior.update(X, y)

    def update_gram_matrix_inverse(self):
        X = np.concatenate(self.X)
        self.K_inv = matrix_inverse(self.kernel.eval(X, X))

    def update_map_covariance(self, f_x: np.ndarray):
        self.cov_map = matrix_inverse(self.neglog_posterior_hessian(f_x))

    def sample_current_approximate_distribution(self, x, n_samples: int = 1):
        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)
        if len(self.X) == 0:
            X = None
        else:
            X = np.concatenate(self.X)
        samples = self.approximate_posterior.sample(x, X, n_samples, K_inv)
        if x.shape[0] == 1:
            samples = samples.item()
        return samples

    def get_mean(self, x):
        if len(self.X) == 0:
            X = None
        else:
            X = np.concatenate(self.X)
        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)

        mean = self.approximate_posterior.get_mean(x, X, K_inv=self.K_inv)
        if x.shape[0] == 1:
            mean = mean.item()
        return mean

    def get_covariance(self, x):
        if len(self.X) == 0:
            X = None
        else:
            X = np.concatenate(self.X)

        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)

        cov = self.approximate_posterior.get_covariance(
            x, X, K_inv=self.K_inv, cov_map=self.cov_map
        )
        if x.shape[0] == 1:
            cov = cov.item()
        return cov
