from typing import List, Tuple, Union

import cvxpy as cp
import numpy as np
from scipy.special import expit

from src.constraints.constraints import EllipticalConstraint
from src.reward_models.logistic_reward_models import (
    LinearLogisticRewardModel,
    LogisticRewardModel,
)
from src.utils import (
    argmax_over_index_set,
    bernoulli_entropy,
    matrix_inverse,
    multivariate_normal_sample,
)


def acquisition_function_random(
    reward_model: LogisticRewardModel,
    candidate_queries: List[np.array],
    return_utility: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray, List]]:
    if return_utility:
        utility = [0] * len(candidate_queries)
        return candidate_queries[np.random.randint(0, len(candidate_queries))], utility
    else:
        return candidate_queries[np.random.randint(0, len(candidate_queries))]


def acquisition_function_bounded_hessian(
    reward_model: LinearLogisticRewardModel,
    candidate_queries: List[np.array],
    return_utility: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray, List]]:
    """Randomly picks a query.

    Args:
        reward_model (LinearLogisticRewardModel): The reward model
        candidate_queries (List[np.array]): The candidate queries.
        return_utility (bool, optional): If the utility for each query should be returned. Defaults to True.

    Returns:
        Union[np.ndarray, Tuple[np.ndarray, List]: Optimal query or (optimal query, utility).
    """

    utility = []
    H_inv = reward_model.hessian_bound_inv
    for x in candidate_queries:
        utility.append((x @ H_inv @ x.T).item())
    argmax = argmax_over_index_set(utility, range(len(candidate_queries)))
    if return_utility:
        return candidate_queries[np.random.choice(argmax)], utility
    else:
        return candidate_queries[np.random.choice(argmax)]


def acquisition_function_optimal_hessian(
    reward_model: LinearLogisticRewardModel,
    candidate_queries: List[np.array],
    theta: np.ndarray,
    return_utility: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray, List]]:
    """Picks the query that minimizes determinant of the hessian inverse an the true parameter.

    Args:
        reward_model (LinearLogisticRewardModel): The reward model
        candidate_queries (List[np.array]): The candidate queries.
        return_utility (bool, optional): If the utility for each query should be returned. Defaults to True.

    Returns:
        Union[np.ndarray, Tuple[np.ndarray, List]: Optimal query or (optimal query, utility).
    """
    utility = []
    for x in candidate_queries:
        H = reward_model.increment_neglog_posterior_hessian(theta, x)
        utility.append(-1 / np.linalg.det(H))
    argmax = argmax_over_index_set(utility, range(len(candidate_queries)))
    if return_utility:
        return candidate_queries[np.random.choice(argmax)], utility
    else:
        return candidate_queries[np.random.choice(argmax)]


def acquisition_function_map_confidence(
    reward_model: LinearLogisticRewardModel,
    candidate_queries: List[np.array],
    alpha: float = 0.7,
    return_utility: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray, List]]:
    utility = []
    mean, covariance = reward_model.get_parameters_moments()
    levelset = -2 * np.log(1 - alpha)
    P = covariance * levelset
    X, _ = reward_model.get_dataset()
    kappas = []

    for x in X:
        theta_i = P @ x.T / np.sqrt(x @ P @ x.T).item() + mean
        kappa_i = (expit(x @ theta_i) * (1 - expit(x @ theta_i))).item()
        kappas.append(kappa_i)

    for x in candidate_queries:
        theta_i = P @ x.T / np.sqrt(x @ P @ x.T) + mean
        kappa_i = (expit(x @ theta_i) * (1 - expit(x @ theta_i))).item()
        kappas.append(kappa_i)
        X.append(x)
        H = reward_model.neglog_posterior_bounded_coordinate_hessian(
            np.concatenate(X), kappas
        )
        utility.append(-1 / np.linalg.det(H).item())
        kappas.pop()
        X.pop()

    argmax = argmax_over_index_set(utility, range(len(candidate_queries)))
    if return_utility:
        return candidate_queries[np.random.choice(argmax)], utility
    else:
        return candidate_queries[np.random.choice(argmax)]


def acquisition_function_bounded_hessian_trace(
    reward_model: LinearLogisticRewardModel,
    candidate_queries: List[np.array],
    return_utility: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray, List]]:
    """Picks the query that minimizes trace of the bounded hessian inverse.

    Args:
        reward_model (LinearLogisticRewardModel): The reward model
        candidate_queries (List[np.array]): The candidate queries.
        return_utility (bool, optional): If the utility for each query should be returned. Defaults to True.

    Returns:
        Union[np.ndarray, Tuple[np.ndarray, List]: Optimal query or (optimal query, utility).
    """
    utility = []
    H_inv = reward_model.hessian_bound_inv
    print("trace: ", np.trace(reward_model.hessian_bound_inv))
    for x in candidate_queries:
        if reward_model.kappa is None:
            kappa = 0.25
        else:
            kappa = reward_model.kappa
        val = np.linalg.norm(H_inv @ x.T) ** 2 / (1 + kappa * x @ H_inv @ x.T)
        utility.append(val.item())
    argmax = argmax_over_index_set(utility, range(len(candidate_queries)))
    if return_utility:
        return candidate_queries[np.random.choice(argmax)], utility
    else:
        return candidate_queries[np.random.choice(argmax)]


def acquisition_function_bounded_coordinate_hessian(
    reward_model: LinearLogisticRewardModel,
    candidate_queries: List[np.array],
    return_utility: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray, List]]:
    """Picks the query that minimizes determinant of the bounded hessian inverse.

    Args:
        reward_model (LinearLogisticRewardModel): The reward model
        candidate_queries (List[np.array]): The candidate queries.
        return_utility (bool, optional): If the utility for each query should be returned. Defaults to True.

    Returns:
        Union[np.ndarray, Tuple[np.ndarray, List]: Optimal query or (optimal query, utility).
    """
    utility = []
    for x in candidate_queries:
        H_inv, _ = reward_model.increment_inv_hessian_coordinate_bound(x)
        utility.append(-np.linalg.det(H_inv))
    argmax = argmax_over_index_set(utility, range(len(candidate_queries)))
    if return_utility:
        return candidate_queries[np.random.choice(argmax)], utility
    else:
        return candidate_queries[np.random.choice(argmax)]


def acquisition_function_map_hessian(
    reward_model: LinearLogisticRewardModel,
    candidate_queries: List[np.array],
    return_utility: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray, List]]:
    """_summary_

    Args:
        reward_model (LinearLogisticRewardModel): _description_
        candidate_queries (List[np.array]): _description_
        return_utility (bool, optional): _description_. Defaults to True.

    Returns:
        Union[np.ndarray, Tuple[np.ndarray, List]: _description_
    """
    utility = []
    for x in candidate_queries:
        utility_y = []
        for y in [1, 0]:
            _, H_inv = reward_model.get_simulated_update(x, y)
            utility_y.append(-np.linalg.det(H_inv))
        utility.append(min(utility_y))

    argmax = argmax_over_index_set(utility, range(len(candidate_queries)))
    if return_utility:
        return candidate_queries[np.random.choice(argmax)], utility
    else:
        return candidate_queries[np.random.choice(argmax)]


def acquisition_function_map_hessian_trace(
    reward_model: LinearLogisticRewardModel,
    candidate_queries: List[np.array],
    return_utility: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray, List]]:
    """_summary_

    Args:
        reward_model (LinearLogisticRewardModel): _description_
        candidate_queries (List[np.array]): _description_
        return_utility (bool, optional): _description_. Defaults to True.

    Returns:
        Union[np.ndarray, Tuple[np.ndarray, List]: _description_
    """
    utility = []
    for x in candidate_queries:
        utility_y = []
        for y in [1, 0]:
            _, H_inv = reward_model.get_simulated_update(x, y)
            utility_y.append(-np.trace(H_inv))
        utility.append(min(utility_y))

    argmax = argmax_over_index_set(utility, range(len(candidate_queries)))
    if return_utility:
        return candidate_queries[np.random.choice(argmax)], utility
    else:
        return candidate_queries[np.random.choice(argmax)]


def acquisition_function_expected_hessian(
    reward_model: LinearLogisticRewardModel,
    candidate_queries: List[np.array],
    n_samples: int = 10,
    return_utility: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray, List]]:
    """_summary_

    Args:
        reward_model (LinearLogisticRewardModel): _description_
        candidate_queries (List[np.array]): _description_
        n_samples (int, optional): _description_. Defaults to 10.
        return_utility (bool, optional): _description_. Defaults to True.

    Returns:
        Union[np.ndarray, Tuple[np.ndarray, List]: _description_
    """
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
    if return_utility:
        return candidate_queries[np.random.choice(argmax)], utility
    else:
        return candidate_queries[np.random.choice(argmax)]


def acquisition_function_bald(
    reward_model: LinearLogisticRewardModel,
    candidate_queries: List[np.array],
    n_samples: int = 50,
    return_utility: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray, List]]:
    """_summary_

    Args:
        reward_model (LinearLogisticRewardModel): _description_
        candidate_queries (List[np.array]): _description_
        n_samples (int, optional): _description_. Defaults to 50.
        return_utility (bool, optional): _description_. Defaults to True.

    Returns:
        Union[np.ndarray, Tuple[np.ndarray, List]: _description_
    """
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
    if return_utility:
        return candidate_queries[np.random.choice(argmax)], utility
    else:
        return candidate_queries[np.random.choice(argmax)]
