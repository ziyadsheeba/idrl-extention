import copy
import json
import multiprocessing
import os
import pickle
import time
from multiprocessing import Pool
from typing import Callable, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import typer
from memory_profiler import profile
from scipy import spatial
from tqdm import tqdm

from src.agents.gp_agent import GPAgent as Agent
from src.constants import DRIVER_PRECOMPUTED_POLICIES_PATH
from src.envs.driver import get_driver_target_velocity
from src.reward_models.kernels import LinearKernel, RBFKernel
from src.reward_models.logistic_reward_models import GPLogisticRewardModel

plt.style.use("ggplot")

from src.nonlinear.driver_config import (
    ALGORITHM,
    CANDIDATE_POLICY_UPDATE_RATE,
    DIMENSIONALITY,
    IDRL,
    N_JOBS,
    NUM_CANDIDATE_POLICIES,
    NUM_QUERY,
    QUERY_LOGGING_RATE,
    SEEDS,
    SIMULATION_STEPS,
    TRAJECTORY_QUERY,
)


def simultate(
    algorithm: str,
    dimensionality: int,
    simulation_steps: int,
    num_candidate_policies: int,
    candidate_policy_update_rate: int,
    query_logging_rate: int,
    num_query: int,
    idrl: bool,
    trajectory_query: bool,
    n_jobs: int,
):
    # true reward parameter
    env = get_driver_target_velocity()
    theta_true = env.reward_w
    optimal_policy = env.get_optimal_policy()

    # Initialize the reward model
    reward_model = GPLogisticRewardModel(
        dim=dimensionality,
        kernel=RBFKernel(dim=dimensionality),
    )

    # Initialize the agents
    agent = Agent(
        query_expert=env.get_comparison_from_full_states,
        get_representation=env.get_full_state,
        get_render_representation=env.get_render_state,
        get_optimal_policy_from_reward_function=env.get_optimal_policy_from_reward_function,
        env_step=env.step,
        env_reset=env.reset,
        precomputed_policy_path=DRIVER_PRECOMPUTED_POLICIES_PATH / "policies.pkl",
        reward_model=reward_model,
        num_candidate_policies=num_candidate_policies,
        idrl=idrl,
        candidate_policy_update_rate=candidate_policy_update_rate,
        representation_space_dim=dimensionality,
        use_trajectories=trajectory_query,
        num_query=num_query,
        n_jobs=n_jobs,
    )

    policy_regret = {}
    cosine_distance = {}
    with tqdm(range(simulation_steps), unit="step") as steps:
        for step in steps:
            estimated_policy = agent.get_mean_optimal_policy()
            r_estimate = env.simulate(estimated_policy)
            r_optimal = env.simulate(optimal_policy)
            r_diff = r_optimal - r_estimate
            policy_regret[step] = r_diff if r_diff > 0 else 0

            mlflow.log_metric("policy_regret", policy_regret[step], step=step)
            steps.set_description(f"Policy Regret {policy_regret[step]}")

            query_best, label, queried_states = agent.optimize_query(
                algorithm=algorithm, n_jobs=1
            )
            agent.update_belief(*query_best, label)

            if step % query_logging_rate == 0:

                env.simulate(estimated_policy)

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


def execute(seed):
    np.random.seed(seed)

    mlflow.set_experiment(f"driver/{ALGORITHM}")
    with mlflow.start_run():
        mlflow.log_param("algorithm", ALGORITHM)
        mlflow.log_param("dimensionality", DIMENSIONALITY)
        mlflow.log_param("canidate_policy_update_rate", CANDIDATE_POLICY_UPDATE_RATE)
        mlflow.log_param("num_candidate_policies", NUM_CANDIDATE_POLICIES)
        mlflow.log_param("idrl", IDRL)
        mlflow.log_param("use_trajectories", TRAJECTORY_QUERY)
        mlflow.log_param("seed", seed)

        simultate(
            algorithm=ALGORITHM,
            dimensionality=DIMENSIONALITY,
            simulation_steps=SIMULATION_STEPS,
            candidate_policy_update_rate=CANDIDATE_POLICY_UPDATE_RATE,
            num_candidate_policies=NUM_CANDIDATE_POLICIES,
            query_logging_rate=QUERY_LOGGING_RATE,
            num_query=NUM_QUERY,
            idrl=IDRL,
            trajectory_query=TRAJECTORY_QUERY,
            n_jobs=N_JOBS,
        )


if __name__ == "__main__":
    for seed in SEEDS:
        execute(10)
