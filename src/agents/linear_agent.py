import pickle
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd

from src.aquisition_functions.aquisition_functions import (
    acquisition_function_bounded_ball_map,
    acquisition_function_bounded_coordinate_hessian,
    acquisition_function_bounded_hessian,
    acquisition_function_current_map_hessian,
    acquisition_function_map_confidence,
    acquisition_function_map_hessian,
    acquisition_function_optimal_hessian,
    acquisition_function_random,
)
from src.reward_models.logistic_reward_models import (
    LinearLogisticRewardModel,
    LogisticRewardModel,
)
from src.utils import get_pairs_from_list, multivariate_normal_sample


class LinearAgent:
    def __init__(
        self,
        query_expert: Callable,
        state_to_features: Callable,
        estimate_state_visitation: Callable,
        get_optimal_policy: Callable,
        get_query_from_policies: Callable,
        precomputed_policy_path: Path,
        reward_model: LogisticRewardModel,
        num_candidate_policies: int,
        idrl: bool,
        candidate_policy_update_rate: int,
        state_space_dim: int,
        use_trajectories: bool,
        num_query: int,
    ):
        """
        Args:
            query_expert (Callable): A function to provide feedback given a query.
            state_to_features (Callable): Transforms states to query features used for the model.
            reward_model (LogisticRewardModel): The reward model.
            state_space_dim (int): The state dimensionality.
        """
        self.state_space_dim = state_space_dim
        self.reward_model = reward_model
        self.query_expert = query_expert
        self.state_to_features = state_to_features
        self.estimate_state_visitation = estimate_state_visitation
        self.get_optimal_policy = get_optimal_policy
        self.get_query_from_policies = get_query_from_policies
        self.idrl = idrl
        self.candidate_policy_update_rate = candidate_policy_update_rate
        self.num_candidate_policies = num_candidate_policies
        self.use_trajectories = use_trajectories
        self.num_query = num_query

        with open(str(precomputed_policy_path), "rb") as f:
            self.precomputed_policies = pickle.load(f)

        self.v = None
        self.counter = 0

    def update_belief(self, x: np.ndarray, y: np.ndarray) -> None:
        self.reward_model.update(x, y)

    def get_parameters_estimate(self):
        return self.reward_model.get_parameters_estimate().squeeze()

    def sample_parameters(self, n_samples: int = 5):
        return self.reward_model.sample_current_approximate_distribution(
            n_samples=n_samples
        )

    def estimate_pairwise_svf_mean(self, policies: list) -> dict:
        svf = []
        for policy in policies:
            svf.append(self.estimate_state_visitation(policy, n_rollouts=1))
        svf = pd.DataFrame(svf).T
        svf = svf.fillna(0)
        states = svf.index.tolist()
        svf = [svf[column].to_numpy() for column in svf.columns]
        svf = get_pairs_from_list(svf)
        svf_diff = [a - b for a, b in svf]
        svf_diff_mean = np.mean(svf_diff, axis=0)
        states = np.array(list(map(np.frombuffer, states)))
        return np.expand_dims(svf_diff_mean, axis=1), states

    def get_candidate_policies(self, use_thompson_sampling: bool = True):
        if use_thompson_sampling:
            sampled_params = self.sample_parameters(
                n_samples=self.num_candidate_policies
            )
            policies = []
            for param in sampled_params:
                policies.append(self.get_optimal_policy(theta=param))

        else:
            raise NotImplementedError()
        return policies

    def get_state_visitation_vector(self):
        print("Recomputing Candidate Policies ...")
        policies = self.get_candidate_policies()
        svf_diff_mean, state_support = self.estimate_pairwise_svf_mean(policies)
        features = [self.state_to_features(x.squeeze().tolist()) for x in state_support]
        features = np.array(features)
        v = features.T @ svf_diff_mean
        return v

    def get_candidate_queries(self):
        if self.use_trajectories:
            if self.num_query > len(self.precomputed_policies):
                raise ValueError(
                    "The number of queries cannot be met. Increase the number of precomputed policies"
                )
            idx = np.random.choice(
                len(self.precomputed_policies), size=self.num_query, replace=False
            )
            _policies = [self.precomputed_policies[i] for i in idx]
            queries = self.get_query_from_policies(_policies, return_trajectories=True)
        else:
            idx = np.random.choice(
                len(self.precomputed_policies),
                size=self.num_query,
                replace=False,
            )
            _policies = [policies[i] for i in idx]
            queries = self.get_query_from_policies(_policies, return_trajectories=False)
            idx = np.random.choice(
                len(queries),
                size=self.num_query,
                replace=False,
            )
            queries = queries[idx, :]

        return queries

    def optimize_query(
        self,
        algorithm: str = "current_map_hessian",
    ) -> Tuple:
        """A function to optimize over queries.

        Args:
            rollout_queries (np.ndarray, optional): The provided candidate queries, trajectories or states.
            If trajectories, assumes an array of shape (episode_length, state_dim, n_trajectories), for states
            (episode_length, state_dim).
            algorithm (str, optional): The algorithm to chose the queries. Defaults to "current_map_hessian".
            v (np.ndarray, optional): The state visitation vector. Defaults to None.
            trajectories (bool, optional): Whether the provided rollout queries are trajectories or states.
            Defaults to False.

        Raises:
            NotImplementedError: If optimization algorithm is not implemented.

        Returns:
            Tuple: The best query as a difference, i.e (feature1-feature2), the label thereof,
            the utility mapping, and the optimal queries in the original space.
        """
        rollout_queries = self.get_candidate_queries()
        if self.idrl and self.counter % self.candidate_policy_update_rate == 0:
            self.v = self.get_state_visitation_vector()

        if self.use_trajectories:
            features = np.apply_along_axis(self.state_to_features, 1, rollout_queries)
            features = LinearLogisticRewardModel.from_trajectories_to_states(features)
        else:
            features = [
                self.state_to_features(x.squeeze().tolist()) for x in rollout_queries
            ]
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
            query_best, utility, argmax = acquisition_function_random(
                self.reward_model, candidate_queries
            )
        elif algorithm == "bounded_coordinate_hessian":
            (
                query_best,
                utility,
                argmax,
            ) = acquisition_function_bounded_coordinate_hessian(
                self.reward_model, candidate_queries, v=self.v
            )
        elif algorithm == "optimal_hessian":
            query_best, utility, argmax = acquisition_function_optimal_hessian(
                self.reward_model, candidate_queries, theta=self.expert.true_parameter
            )
        elif algorithm == "map_confidence":
            query_best, utility, argmax = acquisition_function_map_confidence(
                self.reward_model, candidate_queries
            )
        elif algorithm == "current_map_hessian":
            query_best, utility, argmax = acquisition_function_current_map_hessian(
                self.reward_model, candidate_queries, v=self.v
            )
        elif algorithm == "bounded_ball_map":
            query_best, utility, argmax = acquisition_function_bounded_ball_map(
                self.reward_model, candidate_queries, v=self.v
            )
        else:
            raise NotImplementedError()
        y = self.query_expert(query_best.squeeze().tolist())

        idxs = get_pairs_from_list(range(len(features)))
        queried_idx = idxs[argmax]
        if self.use_trajectories:
            query_best_1 = rollout_queries[:, :, queried_idx[0]].squeeze()
            query_best_2 = rollout_queries[:, :, queried_idx[1]].squeeze()
        else:
            query_best_1 = rollout_queries[queried_idx[0]].squeeze()
            query_best_2 = rollout_queries[queried_idx[1]].squeeze()

        candidate_queries = [x.tobytes() for x in candidate_queries]
        self.counter += 1
        return (
            query_best,
            y,
            dict(zip(candidate_queries, utility)),
            (query_best_1, query_best_2),
        )
