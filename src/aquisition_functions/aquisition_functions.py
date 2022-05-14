from typing import List

import numpy as np

from src.reward_models.logistic_reward_models import LogisticRewardModel
from src.utils import (
    argmax_over_index_set,
    bernoulli_entropy,
    matrix_inverse,
    multivariate_normal_sample,
)


def acquisition_function_random(
    reward_model: LogisticRewardModel, candidate_queries: List[np.array]
) -> np.ndarray:
    """_summary_

    Args:
        reward_model (LogisticRewardModel): The reward model.
        candidate_queries (List): A list of

    Returns:
        np.ndarray: The chosen query.
    """
    return candidate_queries[np.random.randint(0, len(candidate_queries))]


def acquisition_function_bounded_hessian(
    reward_model: LogisticRewardModel, candidate_queries: List[np.array]
) -> np.ndarray:
    utility = []
    # H_inv = reward_model.hessian_bound_inv
    # print(H_inv)
    # print(np.linalg.eigh(H_inv))
    for x in candidate_queries:
        # TO REFACTOR
        H_inv, _ = reward_model.increment_inv_hessian_bound(x)
        utility.append(-np.linalg.det(H_inv))
        # utility.append((x @ H_inv @ x.T).item())

    argmax = argmax_over_index_set(utility, range(len(candidate_queries)))
    return candidate_queries[np.random.choice(argmax)]


def acquisition_function_map_hessian(
    reward_model: LogisticRewardModel, candidate_queries: List[np.array]
) -> np.ndarray:
    utility = []
    for x in candidate_queries:
        utility_y = []
        for y in [1, 0]:
            _, H_inv = reward_model.get_simulated_update(x, y)
            utility_y.append(-np.linalg.det(H_inv))
        utility.append(min(utility_y))

    argmax = argmax_over_index_set(utility, range(len(candidate_queries)))
    return candidate_queries[np.random.choice(argmax)]


def acquisition_function_expected_hessian(
    reward_model: LogisticRewardModel,
    candidate_queries: List[np.array],
    n_samples: int = 10,
) -> np.ndarray:
    utility = []
    for x in candidate_queries:
        label_utility = {}
        for y in [1, 0]:
            mean, H_inv = reward_model.get_simulated_update(x, y)
            samples = multivariate_normal_sample(
                mu=mean, cov=H_inv, n_samples=n_samples
            )
            H_inv = 0
            for sample in samples:
                H = reward_model.neglog_posterior_hessian_increment(theta=sample, x=x)
                H_inv += matrix_inverse(H)
            H_inv = H_inv / len(samples)
            label_utility[y] = -np.linalg.det(H_inv)
        utility.append(label_utility[min(label_utility, key=label_utility.get)])
    argmax = argmax_over_index_set(utility, range(len(candidate_queries)))
    return candidate_queries[np.random.choice(argmax)]


def acquisition_function_bald(
    reward_model: LogisticRewardModel,
    candidate_queries: List[np.array],
    n_samples: int = 400,
) -> np.ndarray:
    utility = []
    for x in candidate_queries:
        p_1, _ = reward_model.get_approximate_predictive_distribution(
            x, method="sampling", n_samples=n_samples
        )
        marginal_entropy = bernoulli_entropy(p_1)

        samples = reward_model.sample_current_approximate_distribution(n_samples)
        expected_entropy = 0
        for i in range(samples.shape[0]):
            sample = samples[i, :]
            p_1 = reward_model.get_likelihood(x, y=1, theta=sample)
            expected_entropy += bernoulli_entropy(p_1)
        expected_entropy = expected_entropy / n_samples
        utility.append(marginal_entropy - expected_entropy)
    argmax = argmax_over_index_set(utility, range(len(candidate_queries)))
    return candidate_queries[np.random.choice(argmax)]
