import pickle
from functools import lru_cache
from itertools import product
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
import tqdm

from src.aquisition_functions.aquisition_functions import (
    acquisition_function_current_map_hessian_gp,
    acquisition_function_predicted_variance,
    acquisition_function_random,
    acquisition_function_variance_ratio,
)
from src.reward_models.logistic_reward_models import (
    GPLogisticRewardModel,
    LogisticRewardModel,
)
from src.utils import (
    bernoulli_log_entropy,
    get_pairs_from_list,
    multivariate_normal_sample,
)


class GPAgent:
    def __init__(
        self,
        query_expert: Callable,
        get_representation: Callable,
        get_render_representation: Callable,
        get_optimal_policy_from_reward_function: Callable,
        env_reset: Callable,
        env_step: Callable,
        precomputed_policy_path: Path,
        reward_model: GPLogisticRewardModel,
        num_candidate_policies: int,
        candidate_policy_update_rate: int,
        representation_space_dim: int,
        idrl: bool = False,
        use_trajectories: bool = False,
        num_query: int = 200,
        n_jobs: int = 1,
        testset_path: Path = None,
    ):
        """
        Args:
            query_expert (Callable): A function to provide feedback given a query.
            get_representation (Callable): Transforms states to query features used for the model.
            reward_model (LogisticRewardModel): The reward model.
            state_space_dim (int): The state dimensionality.
        """
        self.representation_space_dim = representation_space_dim
        self.reward_model = reward_model
        self.query_expert = query_expert
        self.get_representation = get_representation
        self.get_render_representation = get_render_representation
        self.get_optimal_policy_from_reward_function = (
            get_optimal_policy_from_reward_function
        )
        self.reset = env_reset
        self.step = env_step
        self.idrl = idrl
        self.candidate_policy_update_rate = candidate_policy_update_rate
        self.num_candidate_policies = num_candidate_policies
        self.use_trajectories = use_trajectories
        self.num_query = num_query
        self.n_jobs = n_jobs
        self.precomputed_policy_path = precomputed_policy_path
        self.precomputed_policies = None
        self.v = None
        self.counter = 0
        self.testset_path = testset_path
        self.X_test = None
        self.y_test = None

        self.load_precomputed_policies()
        self.load_testset()

    def load_precomputed_policies(self):
        with open(str(self.precomputed_policy_path), "rb") as f:
            self.precomputed_policies = pickle.load(f)

    def load_testset(self):
        if self.testset_path is not None:
            with open(str(self.testset_path / "testset.pkl"), "rb") as f:
                testset = pickle.load(f)
            with open(str(self.testset_path / "query_shape.pkl"), "rb") as f:
                query_shape = pickle.load(f)
            if self.use_trajectories:
                covariates = [
                    np.stack(
                        [
                            np.apply_along_axis(
                                self.get_representation,
                                0,
                                np.frombuffer(a).reshape(query_shape),
                            ),
                            np.apply_along_axis(
                                self.get_representation,
                                0,
                                np.frombuffer(b).reshape(query_shape),
                            ),
                        ],
                        axis=0,
                    )
                    for a, b in testset.keys()
                ]
                self.X_test = np.concatenate(covariates, axis=0)
            else:
                covariates = [
                    np.vstack(
                        [
                            self.get_representation(
                                np.frombuffer(a).reshape(query_shape)
                            ),
                            self.get_representation(
                                np.frombuffer(b).reshape(query_shape)
                            ),
                        ]
                    )
                    for a, b in testset.keys()
                ]
                self.X_test = np.vstack(covariates)
            self.y_test = np.array(list(testset.values()))

    def update_belief(self, x_1: np.ndarray, x_2: np.ndarray, y: np.ndarray) -> None:
        self.reward_model.update(x_1, x_2, y)

    def get_reward_estimate(self, x: np.ndarray):
        return self.reward_model.get_mean(x)

    def get_testset_neglog_likelihood(self):
        if self.testset_path is not None:
            f_test = self.predict_reward(self.X_test)
            log_entropy_test = bernoulli_log_entropy(self.y_test)
            return (
                self.reward_model.neglog_likelihood(f_test, self.y_test)
                - log_entropy_test
            ) / len(self.y_test)
        else:
            raise ValueError()

    def get_testset_accuracy(self, threshold: float = 0.5):
        if self.testset_path is not None:
            f_test = self.predict_reward(self.X_test)
            y_hat = self.reward_model.predict_label(f_test, threshold)
            y_test = (self.y_test >= threshold).astype(int)
            accuracy = (y_hat == y_test).sum() / len(self.y_test)
            return accuracy
        else:
            raise ValueError()

    def predict_reward(self, X):
        return self.reward_model.get_mean(X)

    def sample_reward(self, x, n_samples: int = 1):
        return self.reward_model.sample_current_approximate_distribution(x, n_samples)

    def get_candidate_policies(
        self,
        use_thompson_sampling: bool = True,
    ):
        if use_thompson_sampling:

            reward_functions = [
                lru_cache(maxsize=None)(lambda x: self.sample_reward(np.frombuffer(x)))
                for _ in range(self.num_candidate_policies)
            ]

            policies = []
            pool = Pool(processes=self.n_jobs)
            for policy in tqdm.tqdm(
                pool.starmap(
                    self.get_optimal_policy_from_reward_function,
                    product(reward_functions, self.get_representation),
                ),
                total=len(sampled_params),
            ):
                policies.append(policy)
        else:
            raise NotImplementedError()
        return policies

    def get_mean_optimal_policy(self):
        reward_function = lru_cache(maxsize=None)(
            lambda x: self.get_reward_estimate(np.frombuffer(x))
        )
        return self.get_optimal_policy_from_reward_function(
            reward_function, self.get_representation
        )

    def get_representation_from_policies(
        self, policies: list, n_rollouts: int = 1, return_trajectories: bool = True
    ):
        trajectories = []
        for policy in policies:
            for i in range(n_rollouts):
                done = False
                s = self.reset()
                r = 0
                representations = []
                while not done:
                    a = policy[int(s[-1])]
                    s, reward, done, info = self.step(a)
                    r += reward
                    representations.append(self.get_representation())
                trajectories.append(np.vstack(representations))
        if return_trajectories:
            return np.stack(
                trajectories, axis=2
            ).T  # TODO: Optimize, transposition is slow
        else:
            return np.unique(np.vstack(trajectories), axis=0)

    def get_render_representation_from_policies(
        self, policies: list, n_rollouts: int = 1, return_trajectories: bool = True
    ):
        trajectories = []
        for policy in policies:
            for i in range(n_rollouts):
                done = False
                s = self.reset()
                r = 0
                render_representations = []
                while not done:
                    a = policy[int(s[-1])]
                    s, reward, done, info = self.step(a)
                    r += reward
                    render_representations.append(self.get_render_representation())
                trajectories.append(np.vstack(render_representations))
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
            representations = self.get_representation_from_policies(
                _policies, return_trajectories=True
            )
            render_representations = self.get_render_representation_from_policies(
                _policies, return_trajectories=True
            )
        else:
            idx = np.random.choice(
                len(self.precomputed_policies),
                size=self.num_query,
                replace=False,
            )
            _policies = [self.precomputed_policies[i] for i in idx]
            representations = self.get_representation_from_policies(
                _policies, return_trajectories=False
            )
            render_representations = self.get_render_representation_from_policies(
                _policies, return_trajectories=False
            )
            idx = np.random.choice(
                len(representations),
                size=self.num_query,
                replace=False,
            )
            representations = representations[idx, :]
            render_representations = render_representations[idx, :]
        return representations, render_representations

    def optimize_query(
        self,
        algorithm: str = "current_map_hessian",
        n_jobs: int = 1,
        feedback_mode: str = "sample",
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
        (
            rollout_representations,
            rollout_render_representations,
        ) = self.get_candidate_queries()
        if self.idrl and self.counter % self.candidate_policy_update_rate == 0:
            self.v = self.get_representation_visitation_vector(n_jobs=self.n_jobs)

        rollout_representations = [x for x in rollout_representations]
        representation_pairs = get_pairs_from_list(rollout_representations)
        if algorithm == "random":
            query_best, utility, argmax = acquisition_function_random(
                self.reward_model, representation_pairs, n_jobs=self.n_jobs
            )
        elif algorithm == "predicted_variance":
            query_best, utility, argmax = acquisition_function_predicted_variance(
                self.reward_model, representation_pairs, n_jobs=self.n_jobs
            )
        elif algorithm == "current_map_hessian":
            query_best, utility, argmax = acquisition_function_current_map_hessian_gp(
                self.reward_model, representation_pairs, n_jobs=self.n_jobs
            )
        elif algorithm == "variance_ratio":
            query_best, utility, argmax = acquisition_function_variance_ratio(
                self.reward_model, representation_pairs, n_jobs=self.n_jobs
            )
        else:
            raise NotImplementedError()
        idxs = get_pairs_from_list(range(len(rollout_representations)))
        queried_idx = idxs[argmax]
        if self.use_trajectories:
            render_state_1 = rollout_render_representations[
                :, :, queried_idx[0]
            ].squeeze()
            render_state_2 = rollout_render_representations[
                :, :, queried_idx[1]
            ].squeeze()
        else:
            render_state_1 = rollout_render_representations[queried_idx[0]].squeeze()
            render_state_2 = rollout_render_representations[queried_idx[1]].squeeze()
        y = self.query_expert(*query_best, self.use_trajectories, feedback_mode)
        self.counter += 1
        return (
            query_best,
            y,
            (render_state_1, render_state_2),
        )
