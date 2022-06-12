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
matplotlib.use("Agg")
plt.style.use("ggplot")

from src.linear.driver_config import (
    ALGORITHM,
    CANDIDATE_POLICY_UPDATE_RATE,
    DIMENSIONALITY,
    NUM_CANDIDATE_POLICIES,
    PRIOR_VARIANCE_SCALE,
    QUERY_LOGGING_RATE,
    SIMULATION_STEPS,
    THETA_NORM,
    X_MAX,
    X_MIN,
)


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

    def sample_parameters(self, n_samples: int = 5):
        return self.reward_model.sample_current_approximate_distribution(
            n_samples=n_samples
        )

    def optimize_query(
        self,
        x_min: float,
        x_max: float,
        algorithm: str = "map_hessian_trace",
        v: np.ndarray = None,
        candidate_states: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """_summary_

        Args:
            algortihm (str, optional): _description_. Defaults to "bounded_hessian".
        Returns:
            Tuple[np.ndarray, np.ndarray]: _description_
        """
        if candidate_states is None:
            states = sample_random_cube(
                dim=len(x_min), x_min=x_min, x_max=x_max, n_points=70
            )
        else:
            states = candidate_states
        features = [self.state_to_features(x.squeeze().tolist()) for x in states]
        feature_pairs = get_pairs_from_list(features)
        candidate_queries = [np.expand_dims(a - b, axis=0) for a, b in feature_pairs]

        if algorithm == "bounded_hessian":
            query_best, utility, argmax = acquisition_function_bounded_hessian(
                self.reward_model, candidate_queries
            )
        elif algorithm == "map_hessian":
            query_best, utility, argmax = acquisition_function_map_hessian(
                self.reward_model, candidate_queries
            )
        elif algorithm == "random":
            query_best, utility = acquisition_function_random(
                self.reward_model, candidate_queries
            )
        elif algorithm == "bald":
            query_best, utility, argmax = acquisition_function_bald(
                self.reward_model, candidate_queries
            )
        elif algorithm == "expected_hessian":
            query_best, utility, argmax = acquisition_function_expected_hessian(
                self.reward_model, candidate_queries
            )
        elif algorithm == "bounded_coordinate_hessian":
            (
                query_best,
                utility,
                argmax,
            ) = acquisition_function_bounded_coordinate_hessian(
                self.reward_model, candidate_queries, v=v
            )
        elif algorithm == "map_convex_bound":
            query_best, utility, argmax = acquisition_function_map_convex_bound(
                self.reward_model, candidate_queries
            )
        elif algorithm == "bounded_hessian_trace":
            query_best, utility, argmax = acquisition_function_bounded_hessian_trace(
                self.reward_model, candidate_queries
            )
        elif algorithm == "optimal_hessian":
            query_best, utility, argmax = acquisition_function_optimal_hessian(
                self.reward_model, candidate_queries, theta=self.expert.true_parameter
            )
        elif algorithm == "map_hessian_trace":
            query_best, utility, argmax = acquisition_function_map_hessian_trace(
                self.reward_model, candidate_queries
            )
        elif algorithm == "map_confidence":
            query_best, utility, argmax = acquisition_function_map_confidence(
                self.reward_model, candidate_queries
            )
        elif algorithm == "current_map_hessian":
            query_best, utility, argmax = acquisition_function_current_map_hessian(
                self.reward_model, candidate_queries, v=v
            )
        else:
            raise NotImplementedError()
        y = self.query_expert(query_best.squeeze().tolist())
        self.counter += 1

        idxs = get_pairs_from_list(range(len(features)))
        queried_idx = idxs[argmax]
        state_1 = states[queried_idx[0]].squeeze()
        state_2 = states[queried_idx[1]].squeeze()

        candidate_queries = [x.tobytes() for x in candidate_queries]
        return (
            query_best,
            y,
            dict(zip(candidate_queries, utility)),
            (state_1, state_2),
        )


def simultate(
    algorithm: str,
    dimensionality: int,
    theta_norm: float,
    x_min: float,
    x_max: float,
    prior_variance_scale: float,
    simulation_steps: int,
    num_candidate_policies: int,
    candidate_policy_update_rate: int,
    query_logging_rate: int,
):
    env = get_driver_target_velocity()
    optimal_policy, *_ = env.get_optimal_policy()

    # Initialize the reward model
    reward_model = LinearLogisticRewardModel(
        dim=dimensionality,
        prior_variance=prior_variance_scale * (theta_norm) ** 2 / 2,
        param_norm=theta_norm,
    )
    # Initialize the agents
    agent = Agent(
        query_expert=env.get_comparison_from_feature_diff,
        state_to_features=env.get_query_features,
        reward_model=reward_model,
        state_space_dim=dimensionality,
    )

    for step in tqdm(range(simulation_steps)):

        if step % candidate_policy_update_rate == 0:

            # sample parameters
            if num_candidate_policies > 1:
                params = agent.sample_parameters(n_samples=num_candidate_policies)
            else:
                params = agent.get_parameters_estimate()

            # get optimal policy wrt to each parameter
            policies = []
            for theta in params:
                policy, *_ = env.get_optimal_policy(theta=theta)
                policies.append(policy)

            # get the mean state visitation difference between policies
            svf_diff_mean, states = env.estimate_pairwise_svf_mean(policies)
            features = [env.get_query_features(x.squeeze().tolist()) for x in states]
            features = np.array(features)
            v = features.T @ svf_diff_mean

        query_best, label, utility, queried_states = agent.optimize_query(
            x_min=x_min, x_max=x_max, algorithm=algorithm, candidate_states=states, v=v
        )
        agent.update_belief(query_best, label)

        # compute regret
        env_estimate = get_driver_target_velocity(
            reward_weights=agent.get_parameters_estimate().squeeze()
        )
        estimated_policy, *_ = env_estimate.get_optimal_policy()
        r_estimate = env_estimate.simulate(estimated_policy)
        r_optimal = env_estimate.simulate(optimal_policy)
        mlflow.log_metric("policy_regret", np.abs(r_estimate - r_optimal), step=step)

        if step % query_logging_rate == 0:

            # solve for the mean policy
            theta = agent.get_parameters_estimate().squeeze()
            policy, *_ = env.get_optimal_policy(theta=theta)
            env.simulate(policy)

            # plot the history
            env.plot_history()
            mlflow.log_figure(plt.gcf(), f"driver_{step}.pdf")
            fig_queries = env.plot_query_states_pair(
                queried_states[0], queried_states[1], label
            )

            # log the latest comparison query
            mlflow.log_figure(fig_queries, f"queries_{step}.png")
            plt.close("all")


if __name__ == "__main__":
    mlflow.set_experiment(f"driver/{ALGORITHM}")
    with mlflow.start_run():
        mlflow.log_param("algorithm", ALGORITHM)
        mlflow.log_param("dimensionality", DIMENSIONALITY)
        mlflow.log_param("theta_norm", THETA_NORM)
        mlflow.log_param("x_min", X_MIN)
        mlflow.log_param("x_max", X_MAX)
        mlflow.log_param("prior_variance_scale", PRIOR_VARIANCE_SCALE)
        mlflow.log_param("canidate_policy_update_rate", CANDIDATE_POLICY_UPDATE_RATE)
        mlflow.log_param("num_candidate_policies", NUM_CANDIDATE_POLICIES)

        simultate(
            algorithm=ALGORITHM,
            dimensionality=DIMENSIONALITY,
            theta_norm=THETA_NORM,
            x_min=X_MIN,
            x_max=X_MAX,
            prior_variance_scale=PRIOR_VARIANCE_SCALE,
            simulation_steps=SIMULATION_STEPS,
            candidate_policy_update_rate=CANDIDATE_POLICY_UPDATE_RATE,
            num_candidate_policies=NUM_CANDIDATE_POLICIES,
            query_logging_rate=QUERY_LOGGING_RATE,
        )
