import copy
import multiprocessing
from multiprocessing import Pool
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
    timeit,
)


def acquisition_function_random(
    reward_model: LogisticRewardModel,
    candidate_queries: Union[List[np.ndarray], np.ndarray],
    return_utility: bool = True,
    return_argmax: bool = True,
) -> Union[
    np.ndarray,
    List[Union[np.ndarray, np.ndarray]],
    List[Union[np.ndarray, np.ndarray, int]],
]:
    """Random aquisition function.

    Args:
        reward_model (LogisticRewardModel): The reward model
        candidate_queries (Union[List[np.ndarray], np.ndarray]): The candidate points to evaluate.
        return_utility (bool, optional): Whether or not to return the utility of all points. Defaults to True.
        return_argmax (bool, optional): Whether or not to return the index of the maximum. Defaults to True.

    Returns:
        Union[np.ndarray, List[np.ndarray, np.ndarray], List[np.ndarray, np.ndarray, int]]
    """
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
    candidate_queries: Union[List[np.ndarray], np.ndarray],
    return_utility: bool = True,
    n_jobs: int = 1,
    v: np.ndarray = None,
    return_argmax: bool = True,
) -> Union[
    np.ndarray,
    List[Union[np.ndarray, np.ndarray]],
    List[Union[np.ndarray, np.ndarray, int]],
]:
    """Chooses the queries according to the uniform-bounded hessian determinant.

    Args:
        reward_model (LogisticRewardModel): The reward model
        candidate_queries (Union[List[np.ndarray], np.ndarray]): The candidate points to evaluate.
        return_utility (bool, optional): Whether or not to return the utility of all points. Defaults to True.
        return_argmax (bool, optional): Whether or not to return the index of the maximum. Defaults to True.
        n_jobs (int, optional): Number of jobs to evaluate candidates. Defaults to 1.
        v (np.ndarray, optional): The state-visitation vector. Defaults to None.

    Returns:
        Union[np.ndarray, List[np.ndarray, np.ndarray], List[np.ndarray, np.ndarray, int]]: _description_
    """
    H_inv = reward_model.hessian_bound_inv
    global _get_val

    if v is None:

        def _get_val(x):
            if x.ndim == 1:
                x = np.expand_dims(x, axis=0)
            return (x @ H_inv @ x.T).item()

    else:

        def _get_val(x):
            if x.ndim == 1:
                x = np.expand_dims(x, axis=0)
                kappa = reward_model.kappa
                v_bar = H_inv @ v
                val = (
                    kappa
                    * (x @ v_bar).item() ** 2
                    / (1 + kappa * (x @ H_inv @ x.T).item())
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
    if len(return_vals) == 1:
        return return_vals[0]
    else:
        return return_vals


def acquisition_function_bounded_coordinate_hessian(
    reward_model: LinearLogisticRewardModel,
    candidate_queries: Union[List[np.ndarray], np.ndarray],
    return_utility: bool = True,
    n_jobs: int = 1,
    v: np.ndarray = None,
    return_argmax: bool = True,
) -> Union[
    np.ndarray,
    List[Union[np.ndarray, np.ndarray]],
    List[Union[np.ndarray, np.ndarray, int]],
]:
    """_summary_

    Args:
        reward_model (LogisticRewardModel): The reward model
        candidate_queries (Union[List[np.ndarray], np.ndarray]): The candidate points to evaluate.
        return_utility (bool, optional): Whether or not to return the utility of all points. Defaults to True.
        return_argmax (bool, optional): Whether or not to return the index of the maximum. Defaults to True.
        n_jobs (int, optional): Number of jobs to evaluate candidates. Defaults to 1.
        v (np.ndarray, optional): The state-visitation vector. Defaults to None.

    Returns:
        Union[np.ndarray, List[np.ndarray, np.ndarray], List[np.ndarray, np.ndarray, int]]
    """
    global _get_val
    H_inv = reward_model.hessian_bound_coord_inv

    if v is None:

        def _get_val(x):
            if x.ndim == 1:
                x = np.expand_dims(x, axis=0)
            kappa_i = reward_model.compute_uniform_kappa(x)
            return kappa_i * (x @ H_inv @ x.T).item()

    else:

        def _get_val(x):
            if x.ndim == 1:
                x = np.expand_dims(x, axis=0)
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
    if len(return_vals) == 1:
        return return_vals[0]
    else:
        return return_vals


def acquisition_function_current_map_hessian(
    reward_model: LinearLogisticRewardModel,
    candidate_queries: Union[List[np.ndarray], np.ndarray],
    return_utility: bool = True,
    n_jobs: int = 1,
    v: np.ndarray = None,
    return_argmax: bool = True,
) -> Union[
    np.ndarray,
    List[Union[np.ndarray, np.ndarray]],
    List[Union[np.ndarray, np.ndarray, int]],
]:
    """Uses the determinant of the hessian evaluated at the map estimate to choose queries.

    Args:
        reward_model (LogisticRewardModel): The reward model
        candidate_queries (Union[List[np.ndarray], np.ndarray]): The candidate points to evaluate.
        return_utility (bool, optional): Whether or not to return the utility of all points. Defaults to True.
        return_argmax (bool, optional): Whether or not to return the index of the maximum. Defaults to True.
        n_jobs (int, optional): Number of jobs to evaluate candidates. Defaults to 1.
        v (np.ndarray, optional): The state-visitation vector. Defaults to None.

    Returns:
        Union[np.ndarray, List[np.ndarray, np.ndarray], List[np.ndarray, np.ndarray, int]]
    """
    mean = reward_model.get_parameters_estimate(project=True)
    cov = matrix_inverse(reward_model.neglog_posterior_hessian(theta=mean))
    global _get_val

    if v is None:

        def _get_val(x):
            if x.ndim == 1:
                x = np.expand_dims(x, axis=0)
            kappa_x = (expit(x @ mean) * (1 - expit(x @ mean))).item()
            return kappa_x * (x @ cov @ x.T).item()

    else:

        def _get_val(x):
            if x.ndim == 1:
                x = np.expand_dims(x, axis=0)
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
    if len(return_vals) == 1:
        return return_vals[0]
    else:
        return return_vals


def acquisition_function_optimal_hessian(
    reward_model: LinearLogisticRewardModel,
    candidate_queries: Union[List[np.ndarray], np.ndarray],
    return_utility: bool = True,
    n_jobs: int = 1,
    v: np.ndarray = None,
    return_argmax: bool = True,
) -> Union[
    np.ndarray,
    List[Union[np.ndarray, np.ndarray]],
    List[Union[np.ndarray, np.ndarray, int]],
]:
    """Chooses the queries according to the determinant of the hessian evaluated at the optimal parameter.
    Args:
        reward_model (LogisticRewardModel): The reward model
        candidate_queries (Union[List[np.ndarray], np.ndarray]): The candidate points to evaluate.
        return_utility (bool, optional): Whether or not to return the utility of all points. Defaults to True.
        return_argmax (bool, optional): Whether or not to return the index of the maximum. Defaults to True.
        n_jobs (int, optional): Number of jobs to evaluate candidates. Defaults to 1.
        v (np.ndarray, optional): The state-visitation vector. Defaults to None.

    Raises:
        NotImplementedError: Doesn't support state-visitation vectors yet.

    Returns:
        Union[np.ndarray, List[np.ndarray, np.ndarray], List[np.ndarray, np.ndarray, int]]
    """

    global _get_val

    if v is None:

        def _get_val(x):
            if x.ndim == 1:
                x = np.expand_dims(x, axis=0)
            H = reward_model.increment_neglog_posterior_hessian(theta, x)
            return np.linalg.det(H)

    else:

        def _get_val(x):
            if x.ndim == 1:
                x = np.expand_dims(x, axis=0)
            H = reward_model.increment_neglog_posterior_hessian(theta, x)
            return v.T @ np.linalg.det(H) @ v

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
    if len(return_vals) == 1:
        return return_vals[0]
    else:
        return return_vals


def acquisition_function_map_confidence(
    reward_model: LinearLogisticRewardModel,
    candidate_queries: Union[List[np.ndarray], np.ndarray],
    return_utility: bool = True,
    n_jobs: int = 1,
    v: np.ndarray = None,
    return_argmax: bool = True,
    confidence: float = 0.01,
) -> Union[
    np.ndarray,
    List[Union[np.ndarray, np.ndarray]],
    List[Union[np.ndarray, np.ndarray, int]],
]:
    """Uses the worst case hessian determinant according to approximate confidence sets around
        the current map estimate.

    Args:
        reward_model (LogisticRewardModel): The reward model
        candidate_queries (Union[List[np.ndarray], np.ndarray]): The candidate points to evaluate.
        return_utility (bool, optional): Whether or not to return the utility of all points. Defaults to True.
        return_argmax (bool, optional): Whether or not to return the index of the maximum. Defaults to True.
        n_jobs (int, optional): Number of jobs to evaluate candidates. Defaults to 1.
        v (np.ndarray, optional): The state-visitation vector. Defaults to None.

    Raises:
        NotImplementedError: Doesn' support the state-visitation vector yet.

    Returns:
        Union[np.ndarray, List[np.ndarray, np.ndarray], List[np.ndarray, np.ndarray, int]]
    """

    utility = []
    mean, covariance = reward_model.get_parameters_moments()
    levelset = chi2.ppf(confidence, candidate_queries[0].shape[1])
    P = covariance * levelset
    X, _ = reward_model.get_dataset()

    global _get_kappas

    def _get_kappas(x):
        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)
        theta_i = P @ x.T / np.sqrt(x @ P @ x.T).item() + mean
        kappa_i = (expit(x @ theta_i) * (1 - expit(x @ theta_i))).item()
        return kappa_i

    kappas = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(_get_kappas)(x) for x in X
    )

    global _get_val
    if v is None:

        def _get_val(x, kappas=kappas, X=X):
            if x.ndim == 1:
                x = np.expand_dims(x, axis=0)
            kappas = copy.deepcopy(kappas)
            X = copy.deepcopy(X)
            theta_i = (
                P @ x.T / np.sqrt(x @ P @ x.T).item() + mean
                if np.linalg.norm(x) > 0
                else mean
            )
            kappa_i = (expit(x @ theta_i) * (1 - expit(x @ theta_i))).item()
            kappas.append(kappa_i)
            X.append(x)
            H = reward_model.neglog_posterior_bounded_coordinate_hessian(
                np.concatenate(X), kappas
            )
            return np.linalg.det(H).item()

    else:
        raise NotImplementedError("State visitation vector not supported")

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
    if len(return_vals) == 1:
        return return_vals[0]
    else:
        return return_vals


def acquisition_function_bounded_ball_map(
    reward_model: LinearLogisticRewardModel,
    candidate_queries: Union[List[np.ndarray], np.ndarray],
    return_utility: bool = True,
    n_jobs: int = 1,
    v: np.ndarray = None,
    return_argmax: bool = True,
) -> Union[
    np.ndarray,
    List[Union[np.ndarray, np.ndarray]],
    List[Union[np.ndarray, np.ndarray, int]],
]:
    """Uses a simplified upper bound to choose queries. Please refer to the paper for further details.

    Args:
        reward_model (LogisticRewardModel): The reward model
        candidate_queries (Union[List[np.ndarray], np.ndarray]): The candidate points to evaluate.
        return_utility (bool, optional): Whether or not to return the utility of all points. Defaults to True.
        return_argmax (bool, optional): Whether or not to return the index of the maximum. Defaults to True.
        n_jobs (int, optional): Number of jobs to evaluate candidates. Defaults to 1.
        v (np.ndarray, optional): The state-visitation vector. Defaults to None.

    Returns:
        Union[np.ndarray, List[np.ndarray, np.ndarray], List[np.ndarray, np.ndarray, int]]: _description_
    """

    global _get_val
    H_inv = reward_model.hessian_bound_coord_inv

    if v is None:

        def _get_val(x):
            if x.ndim == 1:
                x = np.expand_dims(x, axis=0)
            kappa_i = reward_model.compute_uniform_kappa(x)
            return round(kappa_i, 7) * round(np.linalg.norm(x), 7)

    else:

        def _get_val(x):
            if x.ndim == 1:
                x = np.expand_dims(x, axis=0)
            kappa_i = round(reward_model.compute_uniform_kappa(x), 7)
            norm = round(np.linalg.norm(x), 7)
            term_1 = 1 + norm * kappa_i
            # term_1 = norm * kappa_i
            # term_2 = (x @ H_inv @ v).item() ** 2
            term_2 = (x @ v).item() ** 2
            return term_2 / term_1

    utility = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(_get_val)(x) for x in candidate_queries
    )
    _argmax = argmax_over_index_set(utility, range(len(candidate_queries)))

    map_candidates = [candidate_queries[i] for i in _argmax]
    query_best, _, argmax_map = acquisition_function_current_map_hessian(
        reward_model, map_candidates
    )
    argmax = _argmax[argmax_map]

    argmax = np.random.choice(argmax)
    return_vals = []
    return_vals.append(query_best)
    if return_utility:
        return_vals.append(utility)
    if return_argmax:
        return_vals.append(argmax)
    if len(return_vals) == 1:
        return return_vals[0]
    else:
        return return_vals


def acquisition_function_map_hessian(
    reward_model: LinearLogisticRewardModel,
    candidate_queries: Union[List[np.ndarray], np.ndarray],
    return_utility: bool = True,
    n_jobs: int = 1,
    v: np.ndarray = None,
    return_argmax: bool = True,
) -> Union[
    np.ndarray,
    List[Union[np.ndarray, np.ndarray]],
    List[Union[np.ndarray, np.ndarray, int]],
]:
    """Uses the determinant of the hessian at the updated map to choose queries.

    Args:
        reward_model (LogisticRewardModel): The reward model
        candidate_queries (Union[List[np.ndarray], np.ndarray]): The candidate points to evaluate.
        return_utility (bool, optional): Whether or not to return the utility of all points. Defaults to True.
        return_argmax (bool, optional): Whether or not to return the index of the maximum. Defaults to True.
        n_jobs (int, optional): Number of jobs to evaluate candidates. Defaults to 1.
        v (np.ndarray, optional): The state-visitation vector. Defaults to None.

    Returns:
        Union[np.ndarray, List[np.ndarray, np.ndarray], List[np.ndarray, np.ndarray, int]]: _description_
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
    if len(return_vals) == 1:
        return return_vals[0]
    else:
        return return_vals
