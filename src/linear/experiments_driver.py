import copy
import json
import multiprocessing
import os
import pickle
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import typer
from scipy import spatial
from tqdm import tqdm

from src.agents.linear_agent import LinearAgent as Agent
from src.constants import DRIVER_PRECOMPUTED_POLICIES_PATH
from src.envs.driver import get_driver_target_velocity
from src.reward_models.logistic_reward_models import (
    LinearLogisticRewardModel,
    LogisticRewardModel,
)

plt.style.use("ggplot")

from src.linear.driver_config import (
    ALGORITHM,
    CANDIDATE_POLICY_UPDATE_RATE,
    DIMENSIONALITY,
    IDRL,
    N_JOBS,
    NUM_CANDIDATE_POLICIES,
    NUM_QUERY,
    PRIOR_VARIANCE_SCALE,
    QUERY_LOGGING_RATE,
    SEEDS,
    SIMULATION_STEPS,
    TESTSET_PATH,
    THETA_NORM,
    TRAJECTORY_QUERY,
    X_MAX,
    X_MIN,
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
    num_query: int,
    idrl: bool,
    trajectory_query: bool,
    n_jobs: int,
    testset_path: Path = None,
):
    # true reward parameter
    env = get_driver_target_velocity()
    theta_true = env.reward_w
    optimal_policy = env.get_optimal_policy()

    # Initialize the reward model
    reward_model = LinearLogisticRewardModel(
        dim=dimensionality,
        prior_variance=prior_variance_scale * (theta_norm) ** 2 / 2,
        param_norm=theta_norm,
        x_min=x_min,
        x_max=x_max,
    )
    # Initialize the agents
    agent = Agent(
        query_expert=env.get_comparison_from_features,
        state_to_features=env.get_reward_features,
        state_to_render_state=env.get_render_state,
        get_optimal_policy=env.get_optimal_policy,
        env_step=env.step,
        env_reset=env.reset,
        precomputed_policy_path=DRIVER_PRECOMPUTED_POLICIES_PATH / "policies.pkl",
        reward_model=reward_model,
        num_candidate_policies=num_candidate_policies,
        idrl=idrl,
        candidate_policy_update_rate=candidate_policy_update_rate,
        feature_space_dim=dimensionality,
        use_trajectories=trajectory_query,
        num_query=num_query,
        n_jobs=n_jobs,
        testset_path=testset_path,
    )

    policy_regret = {}
    cosine_distance = {}
    neglog_likelihood = {}
    accuracy = {}
    with tqdm(range(simulation_steps), unit="step") as steps:
        for step in steps:
            # compute policy_regret and cosine similarity
            theta_hat = agent.get_parameters_estimate().squeeze()
            theta_hat = (
                theta_hat / np.linalg.norm(theta_hat)
                if np.linalg.norm(theta_hat) > 0
                else theta_hat
            )
            estimated_policy = env.get_optimal_policy(theta=theta_hat)
            r_estimate = env.simulate(estimated_policy)
            r_optimal = env.simulate(optimal_policy)
            r_diff = r_optimal - r_estimate
            policy_regret[step] = r_diff if r_diff > 0 else 0
            cosine_distance[step] = (
                spatial.distance.cosine(theta_true, theta_hat)
                if np.linalg.norm(theta_hat) > 0
                else 1
            )
            neglog_likelihood[step] = agent.get_testset_neglog_likelihood()
            accuracy[step] = agent.get_testset_accuracy()

            mlflow.log_metric("policy_regret", policy_regret[step], step=step)
            mlflow.log_metric("cosine_distance", cosine_distance[step], step=step)
            mlflow.log_metric("neglog_likelihood", neglog_likelihood[step], step=step)
            mlflow.log_metric("accuracy", accuracy[step], step=step)
            steps.set_description(f"Policy Regret {policy_regret[step]}")

            query_best, label, queried_states = agent.optimize_query(
                algorithm=algorithm
            )
            agent.update_belief(*query_best, label)
            if step % query_logging_rate == 0:

                # solve for the mean policy
                policy = env.get_optimal_policy(theta=theta_hat)
                env.simulate(policy)

                # plot the history
                env.plot_history()
                mlflow.log_figure(plt.gcf(), f"driver_{step}.pdf")

                # log the latest comparison query
                if trajectory_query:
                    fig_queries = env.plot_query_trajectory_pair(
                        queried_states[0], queried_states[1], label
                    )
                    mlflow.log_figure(fig_queries, f"queries_{step}.png")

                else:
                    fig_queries = env.plot_query_states_pair(
                        queried_states[0], queried_states[1], label
                    )
                    mlflow.log_figure(fig_queries, f"queries_{step}.png")
                plt.close("all")
        mlflow.log_dict(policy_regret, "policy_regret.json")
        mlflow.log_dict(cosine_distance, "cosine_distance.json")
        mlflow.log_dict(neglog_likelihood, "neglog_likelihood.json")
        mlflow.log_dict(accuracy, "accuracy.json")


def execute(seed):
    np.random.seed(seed)

    mlflow.set_experiment(f"driver/linear/{ALGORITHM}")
    with mlflow.start_run():
        mlflow.log_param("algorithm", ALGORITHM)
        mlflow.log_param("dimensionality", DIMENSIONALITY)
        mlflow.log_param("theta_norm", THETA_NORM)
        mlflow.log_param("x_min", X_MIN)
        mlflow.log_param("x_max", X_MAX)
        mlflow.log_param("prior_variance_scale", PRIOR_VARIANCE_SCALE)
        mlflow.log_param("canidate_policy_update_rate", CANDIDATE_POLICY_UPDATE_RATE)
        mlflow.log_param("num_candidate_policies", NUM_CANDIDATE_POLICIES)
        mlflow.log_param("idrl", IDRL)
        mlflow.log_param("use_trajectories", TRAJECTORY_QUERY)
        mlflow.log_param("seed", seed)

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
            num_query=NUM_QUERY,
            idrl=IDRL,
            trajectory_query=TRAJECTORY_QUERY,
            testset_path=TESTSET_PATH,
            n_jobs=N_JOBS,
        )


if __name__ == "__main__":
    for seed in SEEDS:
        execute(seed)
