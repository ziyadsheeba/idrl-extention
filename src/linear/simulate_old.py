import copy
import json
import multiprocessing
import os
import time
from typing import List, Tuple

import cvxpy as cp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import typer
from joblib import Parallel, delayed
from scipy import spatial
from scipy.integrate import quadrature
from scipy.special import expit
from scipy.stats import chi2
from tqdm import tqdm

from src.constants import EXPERIMENTS_PATH
from src.utils import (
    bernoulli_entropy,
    matrix_inverse,
    multivariate_normal_sample,
    sample_random_ball,
    timeit,
)

matplotlib.use("Qt5Agg")

DIMENSIONALITY = 10
STATE_SUPPORT_SIZE = 1000
THETA_UPPER = 2
THETA_LOWER = -2


class Expert:
    def __init__(self, true_parameter: np.ndarray):
        self.true_parameter = true_parameter

    def query_pair_comparison(self, x_1: np.ndarray, x_2: np.ndarray) -> int:
        """_summary_

        Args:
            x_1 (np.ndarray): _description_
            x_2 (np.ndarray): _description_

        Returns:
            int: _description_
        """
        assert isinstance(x_1, np.ndarray) and isinstance(
            x_2, np.ndarray
        ), "Queries must be of type np.ndarray"

        # assert (x_1.shape == self.true_parameter.shape) and (
        #     x_1.shape == self.true_parameter.shape
        # ), "Mismatch between states and parameters dimensions"
        x_delta = x_1 - x_2
        query = 1 if expit(x_delta @ self.true_parameter) >= 0.5 else 0
        return query

    def query_single_absolute_value(self, x: np.ndarray) -> float:
        assert (
            x.shape == self.true_parameter.shape
        ), "Mismatch between the state and parameters dimensions"
        return np.dot(self.true_parameter, x)

    def query_batch_absolute_value(self, X: np.ndarray) -> float:
        assert (
            X.shape[1] == self.true_parameter.shape[0]
        ), "Mismatch between the state and parameters dimensions"
        return X @ self.true_parameter


class Policy:
    def __init__(self, state_support_size: int, state_space_dim: int):
        """_summary_

        Args:
            state_support_size (int): _description_
            state_space_dim (int): _description_
        """
        self.X = np.array(
            [sample_random_ball(state_space_dim) for _ in range(state_support_size)]
        )
        self.visitation_frequencies = np.random.randint(
            low=1, high=1000, size=(state_support_size, 1)
        )
        self.visitation_frequencies = self.visitation_frequencies / np.sum(
            self.visitation_frequencies
        )


class ApproximatePosterior:
    def __init__(
        self,
        weight_space_dim: int,
        prior_variance: float,
        prior_mean: np.ndarray = None,
    ):
        """_summary_

        Args:
            weight_space_dim (int): _description_
            prior_variance (float): _description_
            prior_mean (np.ndarray, optional): _description_. Defaults to None.
        """
        self.weight_space_dim = weight_space_dim
        if prior_mean is None:
            self.prior_mean = np.zeros(weight_space_dim)
        else:
            self.prior_mean = prior_mean

        self.prior_covariance = prior_variance * np.eye(weight_space_dim)
        self.prior_precision = matrix_inverse(self.prior_covariance)

        self.mean = self.prior_mean
        self.precision = self.prior_precision
        self.covariance = matrix_inverse(self.precision)

    def neglog_posterior(self, theta: np.ndarray, y: np.ndarray, X: np.ndarray):
        """_summary_

        Args:
            theta (np.ndarray): _description_
            y (np.ndarray): _description_
            X (np.ndarray): _description_
        """
        eps = 1e-10
        y_hat = expit(X @ theta)
        neg_logprior = 0.5 * (theta - self.prior_mean).dot(self.prior_precision).dot(
            theta - self.prior_mean
        )
        neg_loglikelihood = -np.sum(
            y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps)
        )
        return neg_logprior + neg_loglikelihood

    def neglog_posterior_hessian(self, X: np.ndarray, theta: np.ndarray):
        """_summary_

        Args:
            X (np.ndarray): _description_
            theta (np.ndarray): _description_

        Returns:
            _type_: _description_
        """
        D = np.diag(expit(X @ theta) * (1 - expit(X @ theta)))
        H = X.T @ D @ X + self.prior_covariance
        return H

    def get_likelihood(self, x: np.ndarray, y: int, theta: np.ndarray):
        y_hat = expit(np.dot(x, theta))
        likelihood = y_hat if y == 1 else 1 - y_hat
        return likelihood

    def sample_current_distribution(self, n_samples=1):
        return multivariate_normal_sample(
            mu=self.mean, cov=self.covariance, n_samples=n_samples
        ).T

    def get_approximate_predictive_distribution(
        self, x: np.ndarray, n_samples: int
    ) -> Tuple[float, float]:
        """_summary_

        Args:
            x (np.ndarray): _description_
            n_samples (int): _description_

        Returns:
            Tuple[float, float]: _description_
        """

        """
        # quadrature
        mu = self.mean.T@x
        var = x.T@self.covariance@x
        def fn(f):
            gaussian = (1/np.sqrt(2*np.pi*var)) * np.exp(-0.5*((f-mu)**2)/var)
            return gaussian * expit(f)
        p_1, _ = quadrature(fn, a = -500, b = 500, maxiter = 500) 
        p_0 = 1-p_1
        """
        samples = self.sample_current_distribution(n_samples)
        p_1 = 0
        for i in range(samples.shape[0]):
            sample = samples[i, :]
            p_1 += self.get_likelihood(x=x, y=1, theta=sample)
        p_1 = p_1 / n_samples
        p_0 = 1 - p_1
        return p_1, p_0

    def neglog_posterior_hessian_bound(self, X: np.ndarray, kappa: float) -> np.ndarray:
        """_summary_

        Args:
            X (np.ndarray): _description_
            kappa (float): _description_

        Returns:
            np.ndarray: _description_
        """
        if isinstance(kappa, float):
            H = X.T @ X * kappa + self.prior_covariance
        elif isinstance(kappa, list):
            H = X.T @ np.diag(kappa) @ X + self.prior_covariance
        return H

    def update_approximate_posterior(self, X: np.ndarray, y: np.ndarray) -> None:
        """_summary_

        Args:
            X (np.ndarray): _description_
            y (np.ndarray): _description_
        """
        theta_0 = self.mean
        solution = scipy.optimize.minimize(
            self.neglog_posterior, theta_0, args=(y, X), method="L-BFGS-B"
        )
        print(solution.fun)
        self.mean = solution.x
        self.precision = self.neglog_posterior_hessian(X, self.mean)
        self.covariance = matrix_inverse(self.precision)

    def hallucinate_approximate_posterior(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """_summary_

        Args:
            X (np.ndarray): _description_
            y (np.ndarray): _description_
        """
        theta_0 = self.mean
        solution = scipy.optimize.minimize(
            self.neglog_posterior, theta_0, args=(y, X), method="BFGS"
        )
        mean = solution.x
        precision = self.neglog_posterior_hessian(X, mean)
        return mean, precision


class Agent:
    def __init__(
        self,
        expert: Expert,
        policy: Policy,
        prior_variance: float,
        state_space_dim: int,
        name: str,
    ):
        """_summary_

        Args:
            expert (Expert): _description_
            policy (Policy): _description_
            prior_variance (float): _description_
            state_space_dim (int): _description_
            name (str): _description_
        """
        self.prior_variance = prior_variance
        self.state_space_dim = state_space_dim
        self.memory = []
        self.approximate_posterior = ApproximatePosterior(
            weight_space_dim=state_space_dim, prior_variance=prior_variance
        )
        self.expert = expert
        self.policy = policy

    def update_memory(self, x_1: np.ndarray, x_2: np.ndarray, y: int):
        """_summary_

        Args:
            x_1 (np.ndarray): _description_
            x_2 (np.ndarray): _description_
            y (int): _description_
        """
        self.memory.append((x_1 - x_2, y))

    def update_hallucinated_memory(
        self, hallucinated_memory: List, x_1: np.ndarray, x_2: np.ndarray, y: int
    ) -> List:
        """_summary_

        Args:
            hallucinated_memory (List): _description_
            x_1 (np.ndarray): _description_
            x_2 (np.ndarray): _description_
            y (int): _description_

        Returns:
            List: _description_
        """
        hallucinated_memory.append((x_1 - x_2, y))
        return hallucinated_memory

    def get_dataset(self, memory: List) -> Tuple[np.ndarray, np.ndarray]:
        """_summary_

        Args:
            memory (List): _description_

        Returns:
            Tuple[np.ndarray, np.ndarray]: _description_
        """
        X = np.concatenate([memory[i][0] for i in range(len(memory))], axis=0)
        y = np.array([memory[i][1] for i in range(len(memory))])
        return X, y

    def get_memory_snapshot(self):
        return copy.deepcopy(self.memory)

    def update_belief(self) -> None:
        X, y = self.get_dataset(self.memory)
        self.approximate_posterior.update_approximate_posterior(X, y)

    def hallucinate_update_belief(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        mean, precision = self.approximate_posterior.hallucinate_approximate_posterior(
            X, y
        )
        return mean, precision

    def get_hessian_bound(
        self,
        X: np.ndarray,
        mean: np.ndarray,
        precision: np.ndarray,
        confidence_level: float = 1e-5,
    ) -> float:
        """_summary_

        Args:
            memory (List): _description_
            mean (np.ndarray): _description_
            covariance (np.ndarray): _description_

        Returns:
            float: _description_
        """

        level_set = chi2.ppf(confidence_level, self.state_space_dim)  # DOUBLE CHECK
        P = matrix_inverse(precision) * level_set

        kappa = np.Inf
        theta_opt = None
        kappas = []
        for i in range(X.shape[0]):
            x_i = X[i, :]
            theta_opt_i = (P @ x_i) / (np.sqrt(x_i.T @ P @ x_i)) + mean
            kappa_i = expit(theta_opt_i.T @ x_i) * (1 - expit(theta_opt_i.T @ x_i))
            if kappa_i < kappa:
                kappa = kappa_i
                theta_opt = theta_opt_i
            kappas.append(kappa_i)
        """
        for i in range(X.shape[0]):
            x_i = X[i, :]
            kappas.append(expit(theta_opt.T @ x_i) * (1 - expit(theta_opt.T @ x_i)))
        """
        return kappa, kappas

    def get_uniform_hessian_bound(
        self,
        x: np.ndarray,
    ) -> float:
        """_summary_

        Args:
            x (np.ndarray): _description_

        Returns:
            float: _description_
        """
        theta = cp.Variable(DIMENSIONALITY, 1)
        objective = cp.Maximize(theta.T @ x)
        constraints = [THETA_LOWER <= theta, theta <= THETA_UPPER]
        prob = cp.Problem(objective, constraints)
        result = prob.solve()
        theta_opt = result.value
        kappa = expit(theta_opt.T @ x)
        return kappa

    def get_expected_hessian_inverse(
        self,
        X: np.ndarray,
        mean: np.ndarray,
        covariance: np.ndarray,
        n_samples: int = 200,
    ) -> float:
        # sample from the distribution
        samples = np.random.multivariate_normal(
            mean=mean, cov=covariance, size=n_samples
        )
        H_inv = 0
        for i in range(samples.shape[0]):
            sample = samples[i, :]
            H = self.approximate_posterior.neglog_posterior_hessian(X, sample)
            H_inv += matrix_inverse(H)
        return H_inv / n_samples

    def get_query_cost(self, x_1: np.ndarray, x_2: np.ndarray):
        """_summary_

        Args:
            x_1 (np.ndarray): _description_
            x_2 (np.ndarray): _description_

        Returns:
            _type_: _description_
        """
        f_policy = self.policy.visitation_frequencies
        X_policy = self.policy.X

        values = []
        for y in [1, 0]:
            hallucinated_memory = self.update_hallucinated_memory(
                self.get_memory_snapshot(), x_1, x_2, y
            )
            X, y = self.get_dataset(hallucinated_memory)
            mean, precision = self.hallucinate_update_belief(X, y)
            H_inv = self.get_expected_hessian_inverse(
                X=X, mean=mean, covariance=matrix_inverse(precision)
            )
            values.append(np.linalg.det(H_inv))
            # values.append(np.linalg.det(precision))

            # kappa, kappas = self.get_hessian_bound(X=X, mean=mean, precision=precision)
            # H = self.approximate_posterior.neglog_posterior_hessian_bound(
            #     X, kappa=kappas
            # )

            # #value_1 = f_policy.T @ X_policy @ covariance_1 @ X_policy.T @ f_policy
            # value = 1 / np.linalg.det(H)
            # values.append(value)
        return np.max(values)

    def get_query_cost_uniform(self, x_1: np.ndarray, x_2: np.ndarray):
        """_summary_

        Args:
            x_1 (np.ndarray): _description_
            x_2 (np.ndarray): _description_

        Returns:
            _type_: _description_
        """
        f_policy = self.policy.visitation_frequencies
        X_policy = self.policy.X

        values = []
        for y in [1, 0]:
            hallucinated_memory = self.update_hallucinated_memory(
                self.get_memory_snapshot(), x_1.T, x_2.T, y
            )
            X, y = self.get_dataset(hallucinated_memory)
            mean, precision = self.hallucinate_update_belief(X, y)

            H_inv = self.get_expected_hessian_inverse(
                X=X, mean=mean, covariance=matrix_inverse(precision)
            )
            values.append(np.linalg.det(H_inv))
            # values.append(np.linalg.det(precision))

            # kappa, kappas = self.get_hessian_bound(X=X, mean=mean, precision=precision)
            # H = self.approximate_posterior.neglog_posterior_hessian_bound(
            #     X, kappa=kappas
            # )

            # #value_1 = f_policy.T @ X_policy @ covariance_1 @ X_policy.T @ f_policy
            # value = 1 / np.linalg.det(H)
            # values.append(value)
        return np.max(values)

    def get_query_bald_cost(
        self, x_1: np.ndarray, x_2: np.ndarray, n_samples: int = 500
    ):
        p_1, _ = self.approximate_posterior.get_approximate_predictive_distribution(
            x_1 - x_2, n_samples=n_samples
        )
        marginal_entropy = bernoulli_entropy(p_1)

        samples = self.approximate_posterior.sample_current_distribution(n_samples)
        expected_entropy = 0
        for i in range(samples.shape[0]):
            sample = samples[i, :]
            p_1 = self.approximate_posterior.get_likelihood(
                x_1 - x_2, y=1, theta=sample
            )
            expected_entropy += bernoulli_entropy(p_1)
        expected_entropy = expected_entropy / n_samples
        utility = marginal_entropy - expected_entropy
        return -utility

    def optimize_query(
        self, algortihm: str = "default", query_counts: int = 400
    ) -> Tuple[np.ndarray, np.ndarray]:
        """_summary_

        Args:
            query_counts (int, optional): _description_. Defaults to 100.

        Returns:
            Tuple[np.ndarray, np.ndarray]: _description_
        """

        candidate_queries = [
            (
                sample_random_ball(self.state_space_dim),
                sample_random_ball(self.state_space_dim),
            )
            for _ in range(query_counts)
        ]

        if algortihm == "default":
            results = Parallel(n_jobs=-1, backend="multiprocessing")(
                delayed(self.get_query_cost)(*query) for query in candidate_queries
            )
            # self.get_query_cost(*candidate_queries[0])
            best_query = candidate_queries[np.argmin(results)]
            x_1_best = best_query[0]
            x_2_best = best_query[1]
        # elif algortihm == "bald":
        #     results = Parallel(n_jobs=-1, backend="multiprocessing")(
        #         delayed(self.get_query_bald_cost)(*query) for query in candidate_queries
        #     )
        #     best_query = candidate_queries[np.argmin(results)]
        #     x_1_best = best_query[0]
        #     x_2_best = best_query[1]
        # elif algortihm == "random":
        #     x_1_best = candidate_queries[0][0]
        #     x_2_best = candidate_queries[0][1]
        else:
            raise NotImplementedError(f"Algorithm {algortihm} is not implemented.")

        self.update_memory(
            x_1_best, x_2_best, self.expert.query_pair_comparison(x_1_best, x_2_best)
        )
        self.update_belief()


def simultate(
    num_experiments: int = typer.Option(...), simulation_steps: int = typer.Option(...)
):

    seeds = [np.random.randint(0, 10000) for _ in range(num_experiments)]
    results = {}
    for seed in seeds:

        np.random.seed(seed)

        # Initialize the true parameters of the true reward
        theta = np.random.uniform(
            low=THETA_LOWER, high=THETA_UPPER, size=(DIMENSIONALITY, 1)
        )

        # Initialize the expert
        expert = Expert(true_parameter=theta)

        # Initialize the policy
        policy = Policy(
            state_support_size=STATE_SUPPORT_SIZE, state_space_dim=DIMENSIONALITY
        )

        # Initialize the agents
        agent = Agent(
            expert=expert,
            policy=policy,
            prior_variance=10,
            state_space_dim=DIMENSIONALITY,
            name="Default",
        )
        # bald_agent = Agent(
        #     expert=expert,
        #     policy=policy,
        #     prior_variance=10,
        #     state_space_dim=DIMENSIONALITY,
        #     name="BALD",
        # )
        # random_agent = Agent(
        #     expert=expert,
        #     policy=policy,
        #     prior_variance=10,
        #     state_space_dim=DIMENSIONALITY,
        #     name="Random",
        # )

        results[seed] = {algortihm: [] for algortihm in ["default", "bald", "random"]}

        plt.ion()
        fig, ax = plt.subplots(figsize=(15, 5))
        plt.title("Active query learning")
        plt.xlabel("steps")
        plt.ylabel("cosine similarity")
        ax.set_yscale("log")
        steps = []
        for step in range(simulation_steps):

            agent.optimize_query(algortihm="default")
            # bald_agent.optimize_query(algortihm="bald")
            # random_agent.optimize_query(algortihm="random")

            theta_hat = agent.approximate_posterior.mean
            # theta_hat_random = random_agent.approximate_posterior.mean
            # theta_hat_bald = bald_agent.approximate_posterior.mean

            cosine_distance = spatial.distance.cosine(theta, theta_hat)
            # cosine_distance_random = spatial.distance.cosine(theta, theta_hat_random)
            # cosine_distance_bald = spatial.distance.cosine(theta, theta_hat_bald)

            results[seed]["default"].append(cosine_distance)
            # results[seed]["bald"].append(cosine_distance_bald)
            # results[seed]["random"].append(cosine_distance_random)

            steps.append(step)

            ax.plot(steps, results[seed]["default"], color="green")
            # ax.plot(steps, results[seed]["bald"], color="orange")
            # ax.plot(steps, results[seed]["random"], color="red")

            print(f"Step: {step}")
            print(
                "Cosine Distance: ",
                f"Optimized: {cosine_distance}",
                # f"BALD: {cosine_distance_bald}",
                # f"Random: {cosine_distance_random}",
            )
            plt.pause(0.00001)
            plt.draw()
        os.makedirs(EXPERIMENTS_PATH / "linear", exist_ok=True)
    with open(str(EXPERIMENTS_PATH / "linear" / "results.json"), "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    typer.run(simultate)
