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
from src.reward_models.logistic_reward_models import (
    LinearLogisticRewardModel,
    LogisticRewardModel,
)
from src.utils import (
    from_utility_dict_to_heatmap,
    get_2d_direction_points,
    get_grid_points,
    multivariate_normal_sample,
    sample_random_ball,
    sample_random_cube,
    timeit,
)

matplotlib.use("Qt5Agg")
plt.style.use("ggplot")

from src.linear.active_learning_config import (
    ALGORITHMS,
    DIMENSIONALITY,
    EXPERT_SCALE,
    GRID_RES,
    PLOT,
    PRIOR_VARIANCE_SCALE,
    SEEDS,
    SIMULATION_STEPS,
    THETA_NORM,
    X_MAX,
    X_MIN,
)


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
        p = expit(x_delta @ self.true_parameter).item()
        feedback = np.random.choice([1, 0], p=[p, 1 - p])
        return feedback

    def query_diff_comparison(self, x_delta: np.ndarray) -> int:
        p = expit(x_delta @ self.true_parameter).item()
        feedback = np.random.choice([1, 0], p=[p, 1 - p])
        return feedback

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


class Agent:
    def __init__(
        self,
        expert: Expert,
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
        self.reward_model = reward_model
        self.counter = 0

    def update_belief(self, x: np.ndarray, y: np.ndarray) -> None:
        self.reward_model.update(x, y)

    def get_parameters_estimate(self):
        return self.reward_model.get_parameters_estimate()

    def optimize_query(
        self,
        x_min: float,
        x_max: float,
        grid_res: complex,
        algorithm: str = "map_hessian_trace",
        return_utility: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """_summary_

        Args:
            algortihm (str, optional): _description_. Defaults to "bounded_hessian".
        Returns:
            Tuple[np.ndarray, np.ndarray]: _description_
        """
        candidate_queries = get_grid_points(x_min=x_min, x_max=x_max, n_points=grid_res)

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

        y = self.expert.query_diff_comparison(query_best)
        self.counter += 1
        if return_utility:
            candidate_queries = [x.tobytes() for x in candidate_queries]
            return (
                query_best[0][0],
                query_best[0][1],
                y,
                dict(zip(candidate_queries, utility)),
            )
        else:
            return query_best[0][0], query_best[0][1], y


def simultate(
    algorithm: str,
    dimensionality: int,
    theta_norm: float,
    x_min: float,
    x_max: float,
    grid_res: complex,
    prior_variance_scale: float,
    expert_scale: float,
    plot: bool,
    simulation_steps: int,
) -> list:

    # Initialize the true parameters of the true reward
    theta = np.random.normal(
        loc=0, scale=(theta_norm) ** 2 / 2, size=(dimensionality, 1)
    )

    # Initialize the expert
    expert = Expert(true_parameter=expert_scale * theta)

    # Initialize the reward model
    reward_model = LinearLogisticRewardModel(
        dim=dimensionality,
        prior_variance=prior_variance_scale * (theta_norm) ** 2 / 2,
        param_constraint=SphericalConstraint(
            b=theta_norm**2,
            dim=dimensionality,
        ),
    )

    # Initialize the agents
    agent = Agent(
        expert=expert,
        reward_model=reward_model,
        state_space_dim=dimensionality,
    )

    regret = []
    if plot:
        plt.ion()
        fig, axs = plt.subplots(3, 2, gridspec_kw={"width_ratios": [100, 5]})
        palette = {1: "orange", 0: "pink"}

    steps = []
    queries_x = []
    queries_y = []
    labels = []
    for step in range(simulation_steps):
        query_x, query_y, label, utility = agent.optimize_query(
            x_min=x_min, x_max=x_max, grid_res=grid_res, algorithm=algorithm
        )
        df_query = dict(
            zip(["x", "y", "label"], [queries_x, queries_y, labels]), index=[0]
        )
        theta_hat = agent.get_parameters_estimate()
        cosine_distance = (
            spatial.distance.cosine(theta, theta_hat)
            if np.linalg.norm(theta_hat) > 0
            else 1
        )
        regret.append(cosine_distance)
        steps.append(step)

        queries_x.append(query_x)
        queries_y.append(query_y)
        labels.append(label)
        df = pd.DataFrame(
            dict(zip(["x", "y", "label"], [queries_x, queries_y, labels]))
        )

        if plot:
            axs[0, 0].clear()
            axs[1, 0].clear()
            axs[2, 0].clear()

            # Regret Viz
            axs[0, 0].set_title("Regret")
            axs[0, 0].set_xlabel("Steps")
            axs[0, 0].set_ylabel("Cosine Distance")
            axs[0, 0].set_yscale("log")
            axs[0, 0].plot(steps, regret, color="green")

            # Query viz
            axs[1, 0].set_title("Query Visualization")
            axs[1, 0].set_xlabel("x1")
            axs[1, 0].set_ylabel("x2")
            axs[1, 0].set_ylim(x_min - 0.05, x_max + 0.05)
            axs[1, 0].set_xlim(x_min - 0.05, x_max + 0.05)

            point_1, point_2 = get_2d_direction_points(theta, scale=(x_max - x_min))
            sns.lineplot(
                x=[point_1[0], point_2[0]],
                y=[point_1[1], point_2[1]],
                ax=axs[1, 0],
                color="green",
                label="True Boundary",
            )
            point_1, point_2 = get_2d_direction_points(theta_hat, scale=(x_max - x_min))
            sns.lineplot(
                x=[point_1[0], point_2[0]],
                y=[point_1[1], point_2[1]],
                ax=axs[1, 0],
                color="red",
                label="MAP Boundary",
            )
            sns.scatterplot(
                data=df,
                x="x",
                y="y",
                hue="label",
                palette=palette,
                legend=True,
                ax=axs[1, 0],
            )
            axs[1, 0].plot(
                df.iloc[-1]["x"],
                df.iloc[-1]["y"],
                marker="x",
                color="black",
                markersize=10,
            )

            # Heatmap Viz
            heatmap = from_utility_dict_to_heatmap(utility)
            sns.heatmap(
                heatmap,
                cmap="YlGnBu",
                annot=False,
                ax=axs[2, 0],
                cbar_ax=axs[2, 1],
                vmin=0,
                vmax=1,
                cbar=step == 0,
                annot_kws={"fontsize": 4},
            )
            plt.pause(0.000001)
            plt.draw()

        # Update the agent
        agent.update_belief(np.array([[query_x, query_y]]), label)

        # Log metrics
        print(f"Step: {step}")
        print(
            "Cosine Distance: ",
            f"Optimized: {cosine_distance}",
        )
        print(df["label"].value_counts())
    plt.close("all")
    return regret


if __name__ == "__main__":
    os.makedirs(EXPERIMENTS_PATH, exist_ok=True)
    results = {}
    for seed in SEEDS:
        np.random.seed(seed)
        if seed not in results:
            results[seed] = {}
        for algorithm in ALGORITHMS:
            if algorithm not in results[seed]:
                results[seed][algorithm] = {}
            for dimensionality in DIMENSIONALITY:
                if dimensionality not in results[seed][algorithm]:
                    results[seed][algorithm][dimensionality] = {}

                results[seed][algorithm][dimensionality] = simultate(
                    algorithm=algorithm,
                    dimensionality=dimensionality,
                    theta_norm=THETA_NORM,
                    expert_scale=EXPERT_SCALE,
                    x_min=X_MIN,
                    x_max=X_MAX,
                    grid_res=GRID_RES,
                    prior_variance_scale=PRIOR_VARIANCE_SCALE,
                    plot=PLOT,
                    simulation_steps=SIMULATION_STEPS,
                )
    with open(str(EXPERIMENTS_PATH / "results.json"), "w") as f:
        json.dump(results, f)
