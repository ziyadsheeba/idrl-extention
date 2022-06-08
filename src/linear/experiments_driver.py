import copy
import json
import multiprocessing
import os
import time
from typing import Callable, List, Tuple

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
    acquisition_function_bounded_coordinate_hessian,
    acquisition_function_bounded_hessian,
    acquisition_function_bounded_hessian_trace,
    acquisition_function_current_map_hessian,
    acquisition_function_expected_hessian,
    acquisition_function_map_confidence,
    acquisition_function_map_hessian,
    acquisition_function_map_hessian_trace,
    acquisition_function_optimal_hessian,
    acquisition_function_random,
)
from src.constants import EXPERIMENTS_PATH
from src.constraints.constraints import SphericalConstraint
from src.envs.driver import get_driver_target_velocity
from src.reward_models.logistic_reward_models import (
    LinearLogisticRewardModel,
    LogisticRewardModel,
)
from src.utils import (
    from_utility_dict_to_heatmap,
    get_2d_direction_points,
    get_grid_points,
    get_pairs_from_list,
    multivariate_normal_sample,
    sample_random_cube,
    timeit,
)

matplotlib.use("Qt5Agg")
plt.style.use("ggplot")

DIMENSIONALITY = 8
THETA_NORM = 1
X_LOWER = [-0.7, -0.2, -np.pi, -1, 0]
X_UPPER = [0.7, 0.2, np.pi, 1, 1.456]
GRID_RES = 50j
PRIOR_VARIANCE_SCALE = 1
ALGORITHM = "bounded_hessian"
PLOT = True
SIMULATION_STEPS = 2


class Agent:
    def __init__(
        self,
        query_expert: Callable,
        state_to_features: Callable,
        reward_model: LogisticRewardModel,
        state_space_dim: int,
    ):
        """_summary_

        Args:
            prior_variance (float): _description_
            state_space_dim (int): _description_
            name (str): _description_
        """
        self.state_space_dim = state_space_dim
        self.reward_model = reward_model
        self.query_expert = query_expert
        self.state_to_features = state_to_features
        self.counter = 0

    def update_belief(self, x: np.ndarray, y: np.ndarray) -> None:
        self.reward_model.update(x, y)

    def get_parameters_estimate(self):
        return self.reward_model.get_parameters_estimate()

    def optimize_query(
        self,
        x_min: float,
        x_max: float,
        algorithm: str = "map_hessian_trace",
        return_utility: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """_summary_

        Args:
            algortihm (str, optional): _description_. Defaults to "bounded_hessian".
        Returns:
            Tuple[np.ndarray, np.ndarray]: _description_
        """
        states = sample_random_cube(
            dim=len(x_min), x_min=x_min, x_max=x_max, n_points=70
        )
        features = [self.state_to_features(x.squeeze().tolist()) for x in states]
        feature_pairs = get_pairs_from_list(features)
        candidate_queries = [np.expand_dims(a - b, axis=0) for a, b in feature_pairs]

        # features =
        if algorithm == "bounded_hessian":
            query_best, utility = acquisition_function_bounded_hessian(
                self.reward_model, candidate_queries
            )
        elif algorithm == "map_hessian":
            query_best, utility = acquisition_function_map_hessian(
                self.reward_model, candidate_queries
            )
        elif algorithm == "random":
            query_best, utility = acquisition_function_random(
                self.reward_model, candidate_queries
            )
        elif algorithm == "bald":
            query_best, utility = acquisition_function_bald(
                self.reward_model, candidate_queries
            )
        elif algorithm == "expected_hessian":
            query_best, utility = acquisition_function_expected_hessian(
                self.reward_model, candidate_queries
            )
        elif algorithm == "bounded_coordinate_hessian":
            query_best, utility = acquisition_function_bounded_coordinate_hessian(
                self.reward_model, candidate_queries
            )
        elif algorithm == "map_convex_bound":
            query_best, utility = acquisition_function_map_convex_bound(
                self.reward_model, candidate_queries
            )
        elif algorithm == "bounded_hessian_trace":
            query_best, utility = acquisition_function_bounded_hessian_trace(
                self.reward_model, candidate_queries
            )
        elif algorithm == "optimal_hessian":
            query_best, utility = acquisition_function_optimal_hessian(
                self.reward_model, candidate_queries, theta=self.expert.true_parameter
            )
        elif algorithm == "map_hessian_trace":
            query_best, utility = acquisition_function_map_hessian_trace(
                self.reward_model, candidate_queries
            )
        elif algorithm == "map_confidence":
            query_best, utility = acquisition_function_map_confidence(
                self.reward_model, candidate_queries
            )
        elif algorithm == "current_map_hessian":
            query_best, utility = acquisition_function_current_map_hessian(
                self.reward_model, candidate_queries
            )
        else:
            raise NotImplementedError()
        argmax = np.argmax(utility)
        y = self.query_expert(candidate_queries[argmax])
        self.counter += 1
        if return_utility:
            candidate_queries = [x.tobytes() for x in candidate_queries]
            return (
                query_best,
                y,
                dict(zip(candidate_queries, utility)),
            )
        else:
            return query_best, y


def simultate():
    env = get_driver_target_velocity()
    optimal_policy, *_ = env.get_optimal_policy()

    # Initialize the reward model
    reward_model = LinearLogisticRewardModel(
        dim=DIMENSIONALITY,
        prior_variance=PRIOR_VARIANCE_SCALE * (THETA_NORM) ** 2 / 2,
        param_constraint=SphericalConstraint(
            b=THETA_NORM**2,
            dim=DIMENSIONALITY,
        ),
    )
    # Initialize the agents
    agent = Agent(
        query_expert=env.get_comparison_from_feature_diff,
        state_to_features=env.get_query_features,
        reward_model=reward_model,
        state_space_dim=DIMENSIONALITY,
    )

    for step in range(SIMULATION_STEPS):
        policy, *_ = env.get_optimal_policy(
            theta=reward_model.get_parameters_estimate().squeeze()
        )
        done = False
        s = env.reset()
        r = 0
        print(reward_model.get_parameters_estimate().squeeze())
        while not done:
            a = policy[int(s[-1])]
            s, reward, done, info = env.step(a)
            r += reward
            # env.render("human")
            # time.sleep(0.2)
        query_best, label, utility = agent.optimize_query(
            x_min=X_LOWER, x_max=X_UPPER, algorithm=ALGORITHM
        )
        agent.update_belief(query_best, label)
    env.plot_history()
    plt.savefig("driver.pdf")


if __name__ == "__main__":
    typer.run(simultate)
