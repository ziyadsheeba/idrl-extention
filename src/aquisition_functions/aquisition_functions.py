import copy
from typing import List, Tuple, Union

import cvxpy as cp
import numpy as np
from joblib import Parallel, delayed
from scipy.special import expit
from scipy.stats import chi2

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
    return_argmax: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray, List]]:
    argmax = np.random.randint(0, len(candidate_queries))
    utility = [0] * len(candidate_queries)

    return_vals = []
    return_vals.append(candidate_queries[argmax])
    if return_utility:
        return_vals.append(utility)
    if return_argmax:
        return_vals.append(argmax)
    return return_vals


def acquisition_function_bounded_hessian(
    reward_model: LinearLogisticRewardModel,
    candidate_queries: List[np.array],
    return_utility: bool = True,
    n_jobs: int = 8,
    v: np.ndarray = None,
    return_argmax: bool = True,
) -> Union[np.ndarray, List]:
    """Uses the determinant of the bounded hessian to pick a query. The function is parallelized.

    Args:
        reward_model (LinearLogisticRewardModel): The reward model
        candidate_queries (List[np.array]): The candidate queries.
        return_utility (bool, optional): If the utility for each query should be returned. Defaults to True.
        n_jobs (int, optional): The number of jobs to spawn for the query evaluation.

    Returns:
        Union[np.ndarray, Tuple[np.ndarray, List]: Optimal query or (optimal query, utility).
    """
    H_inv = reward_model.hessian_bound_inv
    global _get_val

    if v is None:

        def _get_val(x):
            return (x @ H_inv @ x.T).item()

    else:
        raise NotImplementedError()

    utility = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(_get_val)(x) for x in candidate_queries
    )
    argmax = argmax_over_index_set(utility, range(len(candidate_queries)))
    argmax = np.random.choice(argmax)

    return_vals = []
    return_vals.append(candidate_queries[argmax])
    if return_utility:
        return_vals.append(utility)
    if return_argmax:
        return_vals.append(argmax)
    return return_vals


def acquisition_function_optimal_hessian(
    reward_model: LinearLogisticRewardModel,
    candidate_queries: List[np.array],
    theta: np.ndarray,
    return_utility: bool = True,
    v: np.ndarray = None,
    n_jobs: int = 8,
    return_argmax: bool = True,
) -> Union[np.ndarray, List]:
    """Picks the query that minimizes determinant of the hessian inverse an the true parameter.

    Args:
        reward_model (LinearLogisticRewardModel): The reward model
        candidate_queries (List[np.array]): The candidate queries.
        return_utility (bool, optional): If the utility for each query should be returned. Defaults to True.

    Returns:
        Union[np.ndarray, Tuple[np.ndarray, List]: Optimal query or (optimal query, utility).
    """
    global _get_val

    if v is None:

        def _get_val(x):
            H = reward_model.increment_neglog_posterior_hessian(theta, x)
            return np.linalg.det(H)

    else:
        raise NotImplementedError()

    utility = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(_get_val)(x) for x in candidate_queries
    )
    argmax = argmax_over_index_set(utility, range(len(candidate_queries)))
    argmax = np.random.choice(argmax)

    return_vals = []
    return_vals.append(candidate_queries[argmax])
    if return_utility:
        return_vals.append(utility)
    if return_argmax:
        return_vals.append(argmax)
    return return_vals


def acquisition_function_map_confidence(
    reward_model: LinearLogisticRewardModel,
    candidate_queries: List[np.array],
    confidence: float = 0.2,
    v: np.ndarray = None,
    return_utility: bool = True,
    return_argmax: bool = True,
) -> Union[np.ndarray, List]:
    """_summary_

    Args:
        reward_model (LinearLogisticRewardModel): _description_
        candidate_queries (List[np.array]): _description_
        confidence (float, optional): _description_. Defaults to 0.2.
        return_utility (bool, optional): _description_. Defaults to True.

    Returns:
        Union[np.ndarray, Tuple[np.ndarray, List]]: _description_
    """
    utility = []
    mean, covariance = reward_model.get_parameters_moments()
    levelset = chi2.ppf(confidence, candidate_queries[0].shape[1])
    P = covariance * levelset
    X, _ = reward_model.get_dataset()

    global _get_kappas

    def _get_kappas(x):
        theta_i = P @ x.T / np.sqrt(x @ P @ x.T).item() + mean
        kappa_i = (expit(x @ theta_i) * (1 - expit(x @ theta_i))).item()
        return kappa_i

    kappas = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(_get_kappas)(x) for x in X
    )

    global _get_val

    if v is None:

        def _get_val(x):
            kappas = copy.deecopy(kappas)
            theta_i = P @ x.T / np.sqrt(x @ P @ x.T).item() + mean
            kappa_i = (expit(x @ theta_i) * (1 - expit(x @ theta_i))).item()
            kappas.append(kappa_i)
            X.append(x)
            H = reward_model.neglog_posterior_bounded_coordinate_hessian(
                np.concatenate(X), kappas
            )
            return np.linalg.det(H).item()

    else:
        raise NotImplementedError()

    utility = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(_get_val)(x) for x in candidate_queries
    )
    argmax = argmax_over_index_set(utility, range(len(candidate_queries)))
    argmax = np.random.choice(argmax)
    return_vals = []
    return_vals.append(candidate_queries[argmax])
    if return_utility:
        return_vals.append(utility)
    if return_argmax:
        return_vals.append(argmax)
    return return_vals


# def acquisition_function_bounded_hessian_trace(
#     reward_model: LinearLogisticRewardModel,
#     candidate_queries: List[np.array],
#     return_utility: bool = True,
#     return_argmax: bool = True,
# ) -> Union[np.ndarray, List]:
#     """Picks the query that minimizes trace of the bounded hessian inverse.

#     Args:
#         reward_model (LinearLogisticRewardModel): The reward model
#         candidate_queries (List[np.array]): The candidate queries.
#         return_utility (bool, optional): If the utility for each query should be returned. Defaults to True.

#     Returns:
#         Union[np.ndarray, Tuple[np.ndarray, List]: Optimal query or (optimal query, utility).
#     """
#     utility = []
#     H_inv = reward_model.hessian_bound_inv
#     for x in candidate_queries:
#         if reward_model.kappa is None:
#             kappa = 0.25
#         else:
#             kappa = reward_model.kappa
#         val = np.linalg.norm(H_inv @ x.T) ** 2 / (1 + kappa * x @ H_inv @ x.T)
#         utility.append(val.item())
#     argmax = argmax_over_index_set(utility, range(len(candidate_queries)))
#     argmax = np.random.choice(argmax)
#     return_vals = []
#     return_vals.append(candidate_queries[argmax])
#     if return_utility:
#         return_vals.append(utility)
#     if return_argmax:
#         return_vals.append(argmax)
#     return return_vals


def acquisition_function_bounded_coordinate_hessian(
    reward_model: LinearLogisticRewardModel,
    candidate_queries: List[np.array],
    v: np.ndarray = None,
    return_utility: bool = True,
    n_jobs: int = 8,
    return_argmax: bool = True,
) -> Union[np.ndarray, List]:
    """Picks the query that minimizes determinant of the bounded hessian inverse.

    Args:
        reward_model (LinearLogisticRewardModel): The reward model
        candidate_queries (List[np.array]): The candidate queries.
        return_utility (bool, optional): If the utility for each query should be returned. Defaults to True.

    Returns:
        Union[np.ndarray, Tuple[np.ndarray, List]: Optimal query or (optimal query, utility).
    """
    global _get_val
    H_inv = reward_model.hessian_bound_coord_inv

    if v is None:

        def _get_val(x):
            kappa_i = reward_model.compute_uniform_kappa(x)
            return kappa_i * (x @ H_inv @ x.T).item()

    else:

        def _get_val(x):
            kappa_i = reward_model.compute_uniform_kappa(x)
            v_bar = H_inv @ v
            val = (
                kappa_i
                * (x @ v_bar).item() ** 2
                / (1 + kappa_i * (x @ H_inv @ x.T).item())
            )
            return val

    utility = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(_get_val)(x) for x in candidate_queries
    )
    argmax = argmax_over_index_set(utility, range(len(candidate_queries)))
    argmax = np.random.choice(argmax)
    return_vals = []
    return_vals.append(candidate_queries[argmax])
    if return_utility:
        return_vals.append(utility)
    if return_argmax:
        return_vals.append(argmax)
    return return_vals


def acquisition_function_map_hessian(
    reward_model: LinearLogisticRewardModel,
    candidate_queries: List[np.array],
    return_utility: bool = True,
    return_argmax: bool = True,
) -> Union[np.ndarray, List]:
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
    argmax = np.random.choice(argmax)
    return_vals = []
    return_vals.append(candidate_queries[argmax])
    if return_utility:
        return_vals.append(utility)
    if return_argmax:
        return_vals.append(argmax)
    return return_vals


def acquisition_function_current_map_hessian(
    reward_model: LinearLogisticRewardModel,
    candidate_queries: List[np.array],
    return_utility: bool = True,
    return_argmax: bool = True,
    n_jobs: int = 8,
    v: np.ndarray = None,
) -> Union[np.ndarray, List]:
    """_summary_

    Args:
        reward_model (LinearLogisticRewardModel): _description_
        candidate_queries (List[np.array]): _description_
        return_utility (bool, optional): _description_. Defaults to True.

    Returns:
        Union[np.ndarray, Tuple[np.ndarray, List]: _description_
    """
    mean, cov = reward_model.get_parameters_moments()
    global _get_val

    if v is None:

        def _get_val(x):
            kappa_x = (expit(x @ mean) * (1 - expit(x @ mean))).item()
            return kappa_x * (x @ cov @ x.T).item()

    else:

        def _get_val(x):
            kappa_x = (expit(x @ mean) * (1 - expit(x @ mean))).item()
            v_bar = cov @ v
            val = (
                kappa_x
                * (x @ v_bar).item() ** 2
                / (1 + kappa_x * (x @ cov @ x.T).item())
            )
            return val

    utility = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(_get_val)(x) for x in candidate_queries
    )
    argmax = argmax_over_index_set(utility, range(len(candidate_queries)))
    argmax = np.random.choice(argmax)
    return_vals = []
    return_vals.append(candidate_queries[argmax])
    if return_utility:
        return_vals.append(utility)
    if return_argmax:
        return_vals.append(argmax)
    return return_vals


def acquisition_function_map_hessian_trace(
    reward_model: LinearLogisticRewardModel,
    candidate_queries: List[np.array],
    return_utility: bool = True,
    return_argmax: bool = True,
) -> Union[np.ndarray, List]:
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
    argmax = np.random.choice(argmax)
    return_vals = []
    return_vals.append(candidate_queries[argmax])
    if return_utility:
        return_vals.append(utility)
    if return_argmax:
        return_vals.append(argmax)
    return return_vals


def acquisition_function_expected_hessian(
    reward_model: LinearLogisticRewardModel,
    candidate_queries: List[np.array],
    n_samples: int = 10,
    return_utility: bool = True,
    return_argmax: bool = True,
) -> Union[np.ndarray, List]:
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
    argmax = np.random.choice(argmax)

    return_vals = []
    return_vals.append(candidate_queries[argmax])
    if return_utility:
        return_vals.append(utility)
    if return_argmax:
        return_vals.append(argmax)
    return return_vals


def acquisition_function_bald(
    reward_model: LinearLogisticRewardModel,
    candidate_queries: List[np.array],
    n_samples: int = 50,
    return_utility: bool = True,
    return_argmax: bool = True,
) -> Union[np.ndarray, List]:
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
    argmax = np.random.choice(argmax)
    return_vals = []
    return_vals.append(candidate_queries[argmax])
    if return_utility:
        return_vals.append(utility)
    if return_argmax:
        return_vals.append(argmax)
    return return_vals
