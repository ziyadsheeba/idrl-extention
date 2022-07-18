import copy
import json
import multiprocessing
import os
import time
from typing import Callable, List, Tuple

import cvxpy as cp
import matplotlib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from scipy import spatial
from scipy.special import expit
from tqdm import tqdm

from src.aquisition_functions.aquisition_functions import (
    acquisition_function_current_map_hessian_gp,
    acquisition_function_predicted_variance,
    acquisition_function_random,
    acquisition_function_variance_ratio,
)
from src.constants import EXPERIMENTS_PATH
from src.reward_models.kernels import RBFKernel
from src.reward_models.logistic_reward_models import (
    GPLogisticRewardModel,
    LogisticRewardModel,
)
from src.utils import (
    from_utility_dict_to_heatmap,
    get_2d_direction_points,
    get_grid_points,
    get_pairs_from_list,
    multivariate_normal_sample,
    sample_random_ball,
    sample_random_cube,
    sample_random_sphere,
    timeit,
)

plt.style.use("ggplot")

from src.nonlinear.active_learning_config import (
    ALGORITHM,
    DIMENSIONALITY,
    KERNEL_PARAMS,
    N_SAMPLES,
    PLOT,
    SEED,
    SIMULATION_STEPS,
    X_MAX,
    X_MIN,
)


class Expert:
    def __init__(self, value_function: Callable):
        self.value_function = value_function

    def query_pair_comparison(self, x_1: np.ndarray, x_2: np.ndarray) -> int:
        assert isinstance(x_1, np.ndarray) and isinstance(
            x_2, np.ndarray
        ), "Queries must be of type np.ndarray"
        val_delta = self.value_function(x_1) - self.value_function(x_2)
        p = expit(val_delta)
        feedback = np.random.choice([1, 0], p=[p, 1 - p])
        return feedback


class Agent:
    def __init__(
        self,
        expert: Expert,
        reward_model: GPLogisticRewardModel,
        state_space_dim: int,
    ):
        """
        Args:
            expert (Expert): An instance of the expert class.
            reward_model (LogisticRewardModel): The reward model.
            state_space_dim (int): The state space dimensions.
        """
        self.state_space_dim = state_space_dim
        self.expert = expert
        self.reward_model = reward_model
        self.counter = 0

    def update_belief(self, x_1: np.ndarray, x_2: np.ndarray, y: np.ndarray) -> None:
        self.reward_model.update(x_1, x_2, y)

    def get_mean(self, x):
        return self.reward_model.get_mean(x)

    def get_covariance(self, x):
        return self.reward_model.get_covariance(x)

    def get_current_neglog_likelihood(self, return_mean=True):
        if return_mean:
            return self.reward_model.get_curret_neglog_likelihood() / self.counter
        else:
            return self.reward_model.get_curret_neglog_likelihood()

    def optimize_query(
        self,
        x_min: float,
        x_max: float,
        n_samples: int,
        algorithm: str = "current_map_hessian",
        return_utility: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Optimizes queries according to the given algorithm.

        Args:
            x_min (float): Minmum state.
            x_max (float): Maximum state.
            n_samples (int): Number of samples to evaluate
            algorithm (str, optional): The algorithm of choice to optimize.
                Defaults to "bounded_coordinate_hessian".
            return_utility (bool, optional): Whether of not to return the utility of each query.
                Defaults to True.

        Raises:
            NotImplementedError: If algorithm is not implemented.

        Returns:
            Tuple
        """
        candidate_queries = np.linspace(x_min, x_max, n_samples)
        candidate_queries = [np.array([[x]]) for x in candidate_queries]
        candidate_queries = get_pairs_from_list(candidate_queries)
        if algorithm == "random":
            query_best, utility, argmax = acquisition_function_random(
                self.reward_model, candidate_queries
            )
        elif algorithm == "current_map_hessian":
            query_best, utility, argmax = acquisition_function_current_map_hessian_gp(
                self.reward_model, candidate_queries, n_jobs=8
            )
        elif algorithm == "predicted_variance":
            query_best, utility, argmax = acquisition_function_predicted_variance(
                self.reward_model, candidate_queries, n_jobs=8
            )
        elif algorithm == "variance_ratio":
            query_best, utility, argmax = acquisition_function_variance_ratio(
                self.reward_model, candidate_queries, n_jobs=8
            )
        else:
            raise NotImplementedError()
        y = self.expert.query_pair_comparison(*query_best)
        self.counter += 1
        return query_best, y, utility


def simultate(
    algorithm: str,
    dimensionality: int,
    x_min: float,
    x_max: float,
    plot: bool,
    simulation_steps: int,
    n_samples: int,
    kernel_params: dict,
) -> list:

    # Define the function that underlies the labeling
    def value_function(x: np.ndarray):
        x = x.squeeze()
        return x**2

    # Initialize the expert
    expert = Expert(value_function=value_function)

    # Initialize the reward model
    reward_model = GPLogisticRewardModel(
        dim=dimensionality,
        kernel=RBFKernel(**kernel_params),
        trajectory=False,
    )

    # Initialize the agents
    agent = Agent(
        expert=expert,
        reward_model=reward_model,
        state_space_dim=dimensionality,
    )
    neglog_likelihood = {}
    plt.plot(figsize=(15, 5))
    for step in tqdm(range(simulation_steps)):
        query, label, utility = agent.optimize_query(
            x_min=x_min, x_max=x_max, n_samples=n_samples, algorithm=algorithm
        )
        query_1 = query[0]
        query_2 = query[1]

        # define the grid points
        points = np.linspace(x_min, x_max, n_samples)

        # plot the underlying value function
        plt.plot(points, [value_function(x) for x in points])

        # plot the gp prediction w/o confidence intervals
        prediction = np.array(
            [agent.get_mean(np.array([[x]])) for x in points]
        ).squeeze()
        var = np.array(
            [[agent.get_covariance(np.array([[x]])) for x in points]]
        ).squeeze()
        plt.plot(points, prediction)
        plt.fill_between(
            points,
            (prediction + 3 * var),
            (prediction - 3 * var),
            color="b",
            alpha=0.1,
        )
        # plot the chosen pairs
        plt.plot(
            query_1,
            agent.get_mean(query_1),
            marker="o",
            markersize=5,
            markerfacecolor="red" if label == 0 else "green",
        )
        plt.plot(
            query_2,
            agent.get_mean(query_2),
            marker="o",
            markersize=5,
            markerfacecolor="green" if label == 0 else "red",
        )

        mlflow.log_figure(plt.gcf(), f"fig_{step}.png")
        plt.cla()
        plt.clf()
        agent.update_belief(*query, label)
        neglog_likelihood[step] = agent.get_current_neglog_likelihood()
        mlflow.log_metric("neglog_likelihood", neglog_likelihood[step], step=step)
        # print(utility)


if __name__ == "__main__":
    np.random.seed(SEED)
    mlflow.set_experiment(f"simple/gp/{ALGORITHM}")
    with mlflow.start_run():
        mlflow.log_param("algorithm", ALGORITHM)
        mlflow.log_param("dimensionality", DIMENSIONALITY)
        mlflow.log_param("x_min", X_MIN)
        mlflow.log_param("x_max", X_MAX)
        mlflow.log_param("n_samples", N_SAMPLES)
        mlflow.log_params(KERNEL_PARAMS)

        simultate(
            algorithm=ALGORITHM,
            dimensionality=DIMENSIONALITY,
            x_min=X_MIN,
            x_max=X_MAX,
            n_samples=N_SAMPLES,
            plot=PLOT,
            simulation_steps=SIMULATION_STEPS,
            kernel_params=KERNEL_PARAMS,
        )
