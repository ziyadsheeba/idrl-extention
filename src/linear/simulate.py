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
import pandas as pd
import seaborn as sns
import typer
from scipy import spatial
from scipy.special import expit
from tqdm import tqdm

from src.aquisition_functions.aquisition_functions import (
    acquisition_function_bald,
    acquisition_function_bounded_hessian,
    acquisition_function_expected_hessian,
    acquisition_function_map_hessian,
    acquisition_function_random,
)
from src.constants import EXPERIMENTS_PATH
from src.constraints.constraints import SimpleConstraint
from src.reward_models.logistic_reward_models import (
    LinearLogisticRewardModel,
    LogisticRewardModel,
)
from src.utils import (
    get_2d_direction_points,
    multivariate_normal_sample,
    sample_random_ball,
    timeit,
)

matplotlib.use("Qt5Agg")

DIMENSIONALITY = 2
STATE_SUPPORT_SIZE = 1000
THETA_UPPER = 1
THETA_LOWER = -1


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

        assert (x_1.shape == self.true_parameter.shape) and (
            x_1.shape == self.true_parameter.shape
        ), "Mismatch between states and parameters dimensions"
        x_delta = x_1 - x_2
        query = 1 if expit(x_delta @ self.true_parameter) >= 0.5 else 0
        return query

    def query_diff_comparison(self, x_delta: np.ndarray) -> int:
        query = 1 if expit(x_delta @ self.true_parameter).item() >= 0.5 else 0
        print(
            expit(x_delta @ self.true_parameter).item(),
        )
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


class Agent:
    def __init__(
        self,
        expert: Expert,
        policy: Policy,
        reward_model: LogisticRewardModel,
        state_space_dim: int,
    ):
        """_summary_

        Args:
            expert (Expert): _description_
            policy (Policy): _description_
            prior_variance (float): _description_
            state_space_dim (int): _description_
            name (str): _description_
        """
        self.state_space_dim = state_space_dim
        self.expert = expert
        self.policy = policy
        self.reward_model = reward_model
        self.counter = 0

    def update_belief(self, x: np.ndarray, y: np.ndarray) -> None:
        self.reward_model.update(x, y)

    def get_parameters_estimate(self):
        return self.reward_model.get_parameters_estimate()

    def optimize_query(
        self, algorithm: str = "map_hessian", query_counts: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """_summary_

        Args:
            algortihm (str, optional): _description_. Defaults to "bounded_hessian".
            query_counts (int, optional): _description_. Defaults to 400.

        Returns:
            Tuple[np.ndarray, np.ndarray]: _description_
        """
        candidate_queries = [
            sample_random_ball(self.state_space_dim, radius=1)
            - sample_random_ball(self.state_space_dim, radius=1)
            for _ in range(query_counts)
        ]

        if algorithm == "bounded_hessian":
            query_best = acquisition_function_bounded_hessian(
                self.reward_model, candidate_queries
            )
            # print(query_best)
        elif algorithm == "map_hessian":
            query_best = acquisition_function_map_hessian(
                self.reward_model, candidate_queries
            )
        elif algorithm == "random":
            query_best = acquisition_function_random(
                self.reward_model, candidate_queries
            )
        elif algorithm == "bald":
            query_best = acquisition_function_bald(self.reward_model, candidate_queries)
        elif algorithm == "expected_hessian":
            query_best = acquisition_function_expected_hessian(
                self.reward_model, candidate_queries
            )
        else:
            raise NotImplementedError()

        y = self.expert.query_diff_comparison(query_best)
        self.update_belief(query_best, y)
        self.counter += 1
        return query_best[0][0], query_best[0][1], y


def simultate(
    num_experiments: int = typer.Option(...), simulation_steps: int = typer.Option(...)
):

    seeds = [np.random.randint(0, 10000) for _ in range(num_experiments)]
    results = {}
    for seed in seeds:
        np.random.seed(seed)

        # Initialize the true parameters of the true reward
        # theta = np.random.normal(
        #     loc=0, scale=(THETA_UPPER - THETA_LOWER) ** 2 / 2, size=(DIMENSIONALITY, 1)
        # )
        theta = np.random.normal(loc=0, scale=1, size=(DIMENSIONALITY, 1))
        # Initialize the expert
        expert = Expert(true_parameter=theta)

        # Initialize the policy
        policy = Policy(
            state_support_size=STATE_SUPPORT_SIZE, state_space_dim=DIMENSIONALITY
        )

        # Initialize the reward model
        reward_model = LinearLogisticRewardModel(
            dim=DIMENSIONALITY,
            prior_variance=(THETA_UPPER - THETA_LOWER) ** 2 / 2,
            param_constraint=SimpleConstraint(
                dim=DIMENSIONALITY, upper=THETA_UPPER, lower=THETA_LOWER
            ),
            kappa=1,
        )

        # Initialize the agents
        agent = Agent(
            expert=expert,
            policy=policy,
            reward_model=reward_model,
            state_space_dim=DIMENSIONALITY,
        )

        results[seed] = {algortihm: [] for algortihm in ["default", "bald", "random"]}

        plt.ion()
        fig, ax = plt.subplots(figsize=(15, 5))
        fig2, ax2 = plt.subplots(figsize=(15, 5))

        plt.title("Active query learning")
        plt.xlabel("steps")
        plt.ylabel("cosine similarity")
        ax.set_yscale("log")
        steps = []
        queries_x = []
        queries_y = []
        labels = []
        palette = {1: "orange", 0: "pink"}

        for step in range(simulation_steps):

            query_x, query_y, label = agent.optimize_query()
            df_query = dict(
                zip(["x", "y", "label"], [queries_x, queries_y, labels]), index=[0]
            )

            theta_hat = agent.get_parameters_estimate()

            cosine_distance = spatial.distance.cosine(theta, theta_hat)

            results[seed]["default"].append(cosine_distance)
            steps.append(step)
            ax.plot(steps, results[seed]["default"], color="green")

            queries_x.append(query_x)
            queries_y.append(query_y)
            labels.append(label)
            df = pd.DataFrame(
                dict(zip(["x", "y", "label"], [queries_x, queries_y, labels]))
            )
            ax2.clear()
            sns.scatterplot(
                data=df,
                x="x",
                y="y",
                hue="label",
                palette=palette,
                legend=True,
                ax=ax2,
            )
            point_1, point_2 = get_2d_direction_points(theta)
            sns.lineplot(
                x=[point_1[0], point_2[0]],
                y=[point_1[1], point_2[1]],
                ax=ax2,
                color="green",
            )
            point_1, point_2 = get_2d_direction_points(
                theta_hat / np.linalg.norm(theta_hat)
            )
            sns.lineplot(
                x=[point_1[0], point_2[0]],
                y=[point_1[1], point_2[1]],
                ax=ax2,
                color="red",
            )

            print(f"Step: {step}")
            print(
                "Cosine Distance: ",
                f"Optimized: {cosine_distance}",
            )
            print(df["label"].value_counts())
            plt.pause(0.00001)
            plt.draw()
        os.makedirs(EXPERIMENTS_PATH / "linear", exist_ok=True)
    with open(str(EXPERIMENTS_PATH / "linear" / "results.json"), "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    typer.run(simultate)
