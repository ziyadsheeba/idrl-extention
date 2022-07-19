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
    LinearLogisticRewardModel,
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
        reward_model: LinearLogisticRewardModel,
        num_candidate_policies: int,
        idrl: bool,
        candidate_policy_update_rate: int,
        feature_space_dim: int,
        use_trajectories: bool,
        num_query: int,
        n_jobs: int = 1,
        testset_path: Path = None,
    ):
        """
        Args:
            query_expert (Callable): A function to provide feedback given a query.
            state_to_features (Callable): Transforms states to query features used for the model.
            reward_model (LogisticRewardModel): The reward model.
            feature_space_dim (int): The state dimensionality.
        """
        self.feature_space_dim = feature_space_dim
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
        self.precomputed_policy_path = precomputed_policy_path
        self.precomputed_policies = None
        self.n_jobs = n_jobs
        self.testset_path = testset_path
        self.X_test = None
        self.y_test = None
        self.v = None
        self.counter = 0

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
                    (
                        np.apply_along_axis(
                            self.state_to_features,
                            0,
                            np.frombuffer(a).reshape(query_shape),
                        ).sum(axis=1)
                        - np.apply_along_axis(
                            self.state_to_features,
                            0,
                            np.frombuffer(b).reshape(query_shape),
                        ).sum(axis=1)
                    )
                    for a, b in testset.keys()
                ]
            else:
                covariates = [
                    self.state_to_features(np.frombuffer(a).reshape(query_shape))
                    - self.state_to_features(np.frombuffer(b).reshape(query_shape))
                    for a, b in testset.keys()
                ]
            self.X_test = np.vstack(covariates)
            self.y_test = np.array(list(testset.values()))

    def update_belief(self, x1: np.ndarray, x2, y: np.ndarray) -> None:
        x = x1 - x2
        self.reward_model.update(x, y)

    def get_parameters_estimate(self):
        return self.reward_model.get_parameters_estimate().squeeze()

    def get_testset_neglog_likelihood(self):
        if self.testset_path is not None:
            theta = self.get_parameters_estimate()
            return self.reward_model.neglog_likelihood(
                theta, self.X_test, self.y_test
            ) / len(self.y_test)
        else:
            raise ValueError()

    def get_testset_accuracy(self, threshold: float = 0.5):
        if self.testset_path is not None:
            theta = self.get_parameters_estimate()
            y_hat = self.reward_model.predict_label(theta, self.X_test, threshold)
            y_test = (self.y_test >= threshold).astype(int)
            accuracy = (y_hat == y_test).sum() / len(self.y_test)
            return accuracy
        else:
            raise ValueError()

    def sample_parameters(self, n_samples: int = 5, method="approximate_posterior"):
        if method == "approximate_posterior":
            return self.reward_model.sample_current_approximate_distribution(
                n_samples=n_samples
            )
        elif method == "mcmc":
            return self.reward_model.sample_mcmc(n_samples=n_samples)

    def estimate_state_visitation(self, policy: np.ndarray, n_rollouts: int = 1):
        svf = {}
        for _ in range(n_rollouts):
            done = False
            s = self.reset()
            while not done:
                a = policy[int(s[-1])]
                s, _, done, _ = self.step(a)
                feature = self.state_to_features()
                feature_arr = np.array(feature).reshape(1, len(feature))
                feature_str = feature_arr.tobytes()
                if feature_str in svf:
                    svf[feature_str] += 1 / n_rollouts
                else:
                    svf[feature_str] = 1 / n_rollouts
        return svf

    def estimate_pairwise_svf_mean(self, policies: list, n_rollouts: int = 1) -> dict:
        svf = [
            self.estimate_state_visitation(policy, n_rollouts=n_rollouts)
            for policy in policies
        ]
        svf = pd.DataFrame(svf).T
        svf = svf.fillna(0)
        features = svf.index.tolist()
        svf = [svf[column].to_numpy() for column in svf.columns]
        svf = get_pairs_from_list(svf)
        svf_diff = [np.abs(a - b) for a, b in svf]
        svf_diff_mean = np.mean(svf_diff, axis=0)
        features = np.vstack(list(map(np.frombuffer, features)))
        return np.expand_dims(svf_diff_mean, axis=1), features

    def get_candidate_policies(self, use_thompson_sampling: bool = True):
        if use_thompson_sampling:
            sampled_params = self.sample_parameters(
                n_samples=self.num_candidate_policies, method="mcmc"
            )
            policies = []
            pool = Pool(processes=self.n_jobs)
            for policy in tqdm.tqdm(
                pool.imap_unordered(self.get_optimal_policy, sampled_params),
                total=len(sampled_params),
            ):
                policies.append(policy)
        else:
            raise NotImplementedError()
        return policies

    def get_state_visitation_vector(self):
        print("Recomputing Candidate Policies ...")
        policies = self.get_candidate_policies()
        svf_diff_mean, features = self.estimate_pairwise_svf_mean(policies)
        v = features.T @ svf_diff_mean
        return v

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
            render_states = self.get_render_state_from_policies(
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
        rollout_features, rollout_render_states = self.get_candidate_queries()
        if self.idrl and self.counter % self.candidate_policy_update_rate == 0:
            self.v = self.get_state_visitation_vector()

        if self.use_trajectories:
            rollout_features = (
                LinearLogisticRewardModel.from_trajectories_to_pseudo_states(
                    rollout_features
                )
            )

        rollout_features = [np.expand_dims(x, axis=0) for x in rollout_features]
        candidate_queries = get_pairs_from_list(rollout_features)

        if algorithm == "bounded_hessian":
            query_best, utility, argmax = acquisition_function_bounded_hessian(
                self.reward_model, candidate_queries, n_jobs=self.n_jobs
            )
        elif algorithm == "map_hessian":
            query_best, utility, argmax = acquisition_function_map_hessian(
                self.reward_model, candidate_queries, n_jobs=self.n_jobs
            )
        elif algorithm == "random":
            query_best, utility, argmax = acquisition_function_random(
                self.reward_model, candidate_queries, n_jobs=self.n_jobs
            )
        elif algorithm == "bounded_coordinate_hessian":
            (
                query_best,
                utility,
                argmax,
            ) = acquisition_function_bounded_coordinate_hessian(
                self.reward_model, candidate_queries, v=self.v, n_jobs=self.n_jobs
            )
        elif algorithm == "optimal_hessian":
            query_best, utility, argmax = acquisition_function_optimal_hessian(
                self.reward_model,
                candidate_queries,
                theta=self.expert.true_parameter,
                n_jobs=self.n_jobs,
            )
        elif algorithm == "map_confidence":
            query_best, utility, argmax = acquisition_function_map_confidence(
                self.reward_model, candidate_queries, n_jobs=self.n_jobs
            )
        elif algorithm == "current_map_hessian":
            query_best, utility, argmax = acquisition_function_current_map_hessian(
                self.reward_model, candidate_queries, v=self.v, n_jobs=self.n_jobs
            )
        elif algorithm == "bounded_ball_map":
            query_best, utility, argmax = acquisition_function_bounded_ball_map(
                self.reward_model, candidate_queries, v=self.v, n_jobs=self.n_jobs
            )
        else:
            raise NotImplementedError()
        y = self.query_expert(*query_best, False, feedback_mode)
        idxs = get_pairs_from_list(range(len(rollout_features)))
        queried_idx = idxs[argmax]
        if self.use_trajectories:
            query_best_1 = rollout_render_states[:, :, queried_idx[0]].squeeze()
            query_best_2 = rollout_render_states[:, :, queried_idx[1]].squeeze()
        else:
            query_best_1 = rollout_render_states[queried_idx[0]].squeeze()
            query_best_2 = rollout_render_states[queried_idx[1]].squeeze()
        self.counter += 1
        return (
            query_best,
            y,
            (query_best_1, query_best_2),
        )
