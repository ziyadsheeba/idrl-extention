import pickle
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
import tqdm

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
    GPLogisticRewardModel,
    LogisticRewardModel,
)
from src.utils import get_pairs_from_list, multivariate_normal_sample


class LinearAgent:
    def __init__(
        self,
        query_expert: Callable,
        state_to_features: Callable,
        state_to_render_state: Callable,
        get_optimal_policy: Callable,
        env_reset: Callable,
        env_step: Callable,
        precomputed_policy_path: Path,
        reward_model: GPLogisticRewardModel,
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
        self.state_to_render_state = state_to_render_state
        self.get_optimal_policy = get_optimal_policy
        self.reset = env_reset
        self.step = env_step
        self.idrl = idrl
        self.candidate_policy_update_rate = candidate_policy_update_rate
        self.num_candidate_policies = num_candidate_policies
        self.use_trajectories = use_trajectories
        self.num_query = num_query

        with open(str(precomputed_policy_path), "rb") as f:
            self.precomputed_policies = pickle.load(f)

        self.v = None
        self.counter = 0

    def update_belief(self, x_1: np.ndarray, x_2: np.ndarray, y: np.ndarray) -> None:
        self.reward_model.update(x_1, x_2, y)

    def get_reward_estimate(self, x: np.ndarray):
        return self.reward_model.get_mean(x).squeeze()

    def sample_reward(self, x, n_samples):
        return self.reward_model.sample_current_approximate_distribution(x, n_samples)

    def get_candidate_policies(
        self, use_thompson_sampling: bool = True, n_jobs: int = 1
    ):
        if use_thompson_sampling:
            sampled_params = self.sample_parameters(
                n_samples=self.num_candidate_policies, method="mcmc"
            )
            policies = []
            pool = Pool(processes=n_jobs)
            for policy in tqdm.tqdm(
                pool.imap_unordered(self.get_optimal_policy, sampled_params),
                total=len(sampled_params),
            ):
                policies.append(policy)
        else:
            raise NotImplementedError()
        return policies

    def get_features_from_policies(
        self, policies: list, n_rollouts: int = 1, return_trajectories: bool = True
    ):
        trajectories = []
        for policy in policies:
            for i in range(n_rollouts):
                done = False
                s = self.reset()
                r = 0
                features = []
                while not done:
                    a = policy[int(s[-1])]
                    s, reward, done, info = self.step(a)
                    r += reward
                    features.append(self.state_to_features())
                trajectories.append(np.vstack(features))
        if return_trajectories:
            return np.stack(trajectories, axis=2)
        else:
            return np.unique(np.vstack(trajectories), axis=0)

    def get_render_state_from_policies(
        self, policies: list, n_rollouts: int = 1, return_trajectories: bool = True
    ):
        trajectories = []
        for policy in policies:
            for i in range(n_rollouts):
                done = False
                s = self.reset()
                r = 0
                render_states = []
                while not done:
                    a = policy[int(s[-1])]
                    s, reward, done, info = self.step(a)
                    r += reward
                    render_states.append(self.state_to_render_state())
                trajectories.append(np.vstack(render_states))
        if return_trajectories:
            return np.stack(trajectories, axis=2)
        else:
            return np.unique(np.vstack(trajectories), axis=0)

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
            features = self.get_features_from_policies(
                _policies, return_trajectories=True
            )
            render_states = self.get_render_state_from_policies(
                _policies, return_trajectories=True
            )
        else:
            idx = np.random.choice(
                len(self.precomputed_policies),
                size=self.num_query,
                replace=False,
            )
            _policies = [self.precomputed_policies[i] for i in idx]
            features = self.get_features_from_policies(
                _policies, return_trajectories=False
            )
            render_states = get_render_state_from_policies(
                _policies, return_trajectories=False
            )
            idx = np.random.choice(
                len(features),
                size=self.num_query,
                replace=False,
            )
            features = features[idx, :]
            render_states = render_states[idx, :]
        return features, render_states

    def optimize_query(
        self,
        algorithm: str = "current_map_hessian",
        n_jobs: int = 1,
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
        rollout_features, rollout_render_states = self.get_candidate_queries()
        if self.idrl and self.counter % self.candidate_policy_update_rate == 0:
            self.v = self.get_state_visitation_vector(n_jobs=n_jobs)

        if self.use_trajectories:
            rollout_features = (
                LinearLogisticRewardModel.from_trajectories_to_pseudo_states(
                    rollout_features
                )
            )

        rollout_features = [x for x in rollout_features]
        feature_pairs = get_pairs_from_list(rollout_features)
        candidate_queries = [np.expand_dims(a - b, axis=0) for a, b in feature_pairs]
        if algorithm == "random":
            query_best, utility, argmax = acquisition_function_random(
                self.reward_model, candidate_queries, n_jobs=n_jobs
            )
        else:
            raise NotImplementedError()
        y = self.query_expert(query_best.squeeze().tolist())

        idxs = get_pairs_from_list(range(len(rollout_features)))
        queried_idx = idxs[argmax]
        if self.use_trajectories:
            query_best_1 = rollout_render_states[:, :, queried_idx[0]].squeeze()
            query_best_2 = rollout_render_states[:, :, queried_idx[1]].squeeze()
        else:
            query_best_1 = rollout_render_states[queried_idx[0]].squeeze()
            query_best_2 = rollout_render_states[queried_idx[1]].squeeze()

        candidate_queries = [x.tobytes() for x in candidate_queries]
        self.counter += 1
        return (
            query_best,
            y,
            dict(zip(candidate_queries, utility)),
            (query_best_1, query_best_2),
        )
