import os
import subprocess
import time
from functools import wraps
from typing import Any, Iterable, List, Optional, Set, Union

import cv2
import numpy as np
import wrapt
from numpy import linalg, random
from scipy.linalg.lapack import dpotrf, dpotri


def sample_random_ball(dimension: int, radius: int = 1):

    """Generates a random vector on a unit ball

    Args:
        dimension (int): _description_
        radius (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    random_directions = random.normal(size=(1, dimension))
    random_directions /= linalg.norm(random_directions, axis=0)
    random_radii = random.random() ** (1 / dimension)
    return radius * (random_directions * random_radii)


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Function {func.__name__} took {total_time:.4f} seconds")
        return result

    return timeit_wrapper


def matrix_inverse(A: np.ndarray) -> np.ndarray:
    """Computes the inverse of a matrix

    Args:
        A (np.ndarray): A real non-singular matrix

    Returns:
        np.ndarray: The inverse of matrix A.
    """
    z, _ = dpotrf(A, False, False)
    A_inv, info = dpotri(z)
    if info > 0:
        raise linalg.LinAlgError("Singular matrix")
    A_inv = np.triu(A_inv) + np.triu(A_inv, k=1).T
    return A_inv


def bernoulli_entropy(bias: float):
    eps = 1e-10
    if bias > 1 or bias < 0:
        raise ValueError("The bias value must be in the range (0, 1)")
    return -bias * np.log(bias + eps) - (1 - bias) * np.log(1 - bias + eps)


def subsample_sequence(old_len, new_len):
    if new_len == old_len:
        return 0, old_len
    assert new_len < old_len
    max_start = old_len - new_len
    start = np.random.randint(0, max_start)
    return start, start + new_len


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def get_array_from_scalar(x, dtype=None):
    x = np.array(x)
    if np.ndim(x) == 0:
        x = np.array([x])
    if dtype is not None:
        x = dtype(x)
    return x


def uniformly_sample_evaluation_states(
    env, n_samples, min_val=-10, max_val=10, noise_var=0, **reward_params
):
    n_dim = env.Ndim_repr
    x_test = min_val + (max_val - min_val) * np.random.sample((n_samples, n_dim))
    y_test = []

    for x_val in x_test:
        reward = np.dot(env.reward_w, x_val)
        reward += np.random.normal(0, noise_var)
        y_test.append(reward)
    return get_unique_x_y(x_test, y_test)


def get_rollouts_data(env, n_rollouts, policy=None, eps=0, noise_var=0):
    data = []
    for _ in range(n_rollouts):
        obs = env.reset()
        done = False
        while not done:
            if policy is None or np.random.random() < eps:
                action = env.action_space.sample()
            else:
                action = policy.get_action(obs)
            obs, reward, done, info = env.step(action)
            reward += np.random.normal(0, noise_var)
            data.append((info["gp_repr"], reward))
    return data


def np_to_tuple(x):
    """Convert a numpy array to a tuple, and converte a 0-d array to a 1-d tuple."""
    return tuple(np.atleast_1d(x).tolist())


def get_unique_x_y(x, y):
    _, indices = np.unique(x, axis=0, return_index=True)
    x, y = np.array(x)[indices], np.array(y)[indices]
    return x, y


def pdf_multivariate_gauss(x, mu, cov):
    """
    Caculate the multivariate normal density (pdf)
    Keyword arguments:
        x = numpy array of a "d" sample vector
        mu = numpy array of a "d" mean vector
        cov = "numpy array of a d x d" covariance matrix
    """
    d = mu.shape[0]
    assert mu.shape == (d,)
    assert x.shape == (d,)
    assert cov.shape == (d, d)
    mu = np.reshape(mu, (d, 1))
    x = np.reshape(x, (d, 1))
    part1 = (2 * np.pi) ** (-d / 2) * np.linalg.det(cov) ** (-1 / 2)
    part2 = (-1 / 2) * ((x - mu).T.dot(np.linalg.inv(cov))).dot((x - mu))
    return float(part1 * np.exp(part2))


def multivariate_normal_sample(
    mu: np.ndarray, cov: np.ndarray, n_samples: int = 1
) -> np.ndarray:
    """Uses the Cholesky decomposition of the covariance matrix which provides
        a significant speedup over the standard sampling that numpy uses.

    Args:
        mu (np.ndarray): mean vector.
        cov (np.ndarray): covariance matrix.
        n_samples (int, optional): number of samples to generate. Defaults to 1.

    Returns:
        np.ndarray: a matrix of shape (dim, n_samples), where dim is the dimensionality
            of the sample space.
    """
    A = np.linalg.cholesky(cov)
    z = np.random.randn(mu.shape[0], n_samples)
    if len(mu.shape) == 1:
        mu = np.expand_dims(mu, 1)
    mu_stacked = np.repeat(mu, n_samples, axis=1)
    samples = mu_stacked + A @ z
    return samples.T


def get_hash(x):
    if isinstance(x, np.ndarray):
        return hash(x.tostring())
    else:
        return hash(str(x))


def capitalize_underscore(s):
    sl = s.split("_")
    sl = [s.capitalize() for s in sl]
    s = "_".join(sl)
    return s


def gaussian_log_likelihood(
    y: np.ndarray, mu: np.ndarray, cov: np.ndarray
) -> Optional[float]:
    assert len(y.shape) == 1
    n = y.shape[0]
    assert mu.shape == (n,)
    assert cov.shape == (n, n)
    try:
        A = np.linalg.cholesky(cov)
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return None
    cov_inv = np.matmul(A_inv.T, A_inv)
    ymu = y - mu
    logP = (
        -n / 2 * np.log(2 * np.pi)
        - 1 / 2 * np.log(np.linalg.det(cov))
        - 1 / 2 * np.dot(ymu.T, np.dot(cov_inv, ymu))
    )
    return logP


def mean_jaccard_distance(sets: List[Set[Any]]) -> float:
    """
    Compute the mean Jaccard distance for sets A_1, \dots A_n:
        d = \frac{1}{n} \sum_{i=1}^{n-1} \sum_{j=i+1}^n (1 - J(A_i, A_j))
    where J(A, B) is the Jaccard index between sets A and B and 1-J(A, B)
    is the Jaccard distance.
    """
    n = len(sets)
    assert n > 0
    if n == 1:
        return 0
    else:
        d = 0.0
        for i in range(n - 1):
            for j in range(i + 1, n):
                d += 1 - jaccard_index(sets[i], sets[j])
        d /= n * (n - 1) / 2
        return d


def jaccard_index(A: Set[Any], B: Set[Any]) -> float:
    """
    Compute the Jaccard index between two sets A and B:
        J(A, B) = \frac{|A \cap B|}{|A \cup B|}
    """
    return len(A.intersection(B)) / len(A.union(B))


def timing(f):
    @wrapt.decorator
    def wrapper(wrapped, instance, args, kwargs):
        start = time.time()
        result = wrapped(*args, **kwargs)
        end = time.time()
        delta = end - start

        if not hasattr(instance, "timing") or instance.timing is None:
            instance.timing = dict()
        funcname = wrapped.__name__
        if funcname not in instance.timing:
            instance.timing[funcname] = 0
        instance.timing[funcname] += delta

        return result

    return wrapper(f)


def get_deterministic_policy_matrix(
    deterministic_policy: Union[List, np.ndarray], N_actions: int
) -> np.ndarray:
    """
    Takes a deterministic policy that contains an action per state and returns
    the encoding as a (N_states, N_actions) matrix, i.e. 1 for the deterministic
    action in each state and zero otherwise.
    """
    assert isinstance(deterministic_policy, list) or (
        isinstance(deterministic_policy, np.ndarray)
        and len(deterministic_policy.shape) == 1
    )
    N_states = len(deterministic_policy)
    policy = np.zeros((N_states, N_actions), dtype=np.float32)
    policy[np.arange(N_states), deterministic_policy] = 1
    return policy


def get_dict_default(dictionary, key, default):
    if key in dictionary:
        return dictionary[key]
    else:
        return default


def get_dict_assert(dictionary, key):
    assert key in dictionary
    return dictionary[key]


def argmax_over_index_set(lst: List[float], index_set: Iterable[int]) -> List[int]:
    """
    Computes the argmax function over a subset of a list.
    In contrast to the related `np.argmax` function, this returns a list of all
    indices that have the maximal value.
    Args:
        lst (List): Find the argmax of the values in this list
        index_set (Iterabel): Restricts the list to the indices in this set
    Returns:
        List: a list of elements in index_set at which lst has the maximal value
    """
    max_idx = [0]
    max_val = -float("inf")
    for i in index_set:
        if lst[i] > max_val:
            max_val = lst[i]
            max_idx = [i]
        elif lst[i] == max_val:
            max_idx.append(i)
    return max_idx


def get_acquisition_function_label(acquisition_function, latex=False):
    label = acquisition_function["label"]

    if (
        "use_thompson_sampling_for_candidate_policies" in acquisition_function
        and acquisition_function["use_thompson_sampling_for_candidate_policies"]
        and "update_candidate_policies_every_n" in acquisition_function
        and acquisition_function["update_candidate_policies_every_n"] is not None
        and "n_candidate_policies_thompson_sampling" in acquisition_function
        and acquisition_function["n_candidate_policies_thompson_sampling"] is not None
        and label != "2s_ts"
    ):
        label += "__TS_{}_every_{}".format(
            acquisition_function["n_candidate_policies_thompson_sampling"],
            acquisition_function["update_candidate_policies_every_n"],
        )

    if (
        "observation_batch_size" in acquisition_function
        and acquisition_function["observation_batch_size"]
    ):
        label += "_bs_" + str(acquisition_function["observation_batch_size"])

    if (
        "fix_candidate_queries" in acquisition_function
        and acquisition_function["fix_candidate_queries"]
    ):
        label += "_fixed_" + str(acquisition_function["get_evaluation_set_from"])

    if (
        "simple_model" in acquisition_function
        and acquisition_function["simple_model"]
        and "uncertainty_p" in acquisition_function
    ):
        p = acquisition_function["uncertainty_p"]
        label += f"_p_{p}"

    if latex:
        label = label.replace("_", "\_")
    return label


def get_acquisition_function_label_clean(acquisition_function, latex=False):
    label = acquisition_function["label"]
    if label == "idrl":
        if latex:
            clean_label = "\\texttt{IDRL}"
        else:
            clean_label = "IDRL"
    else:
        clean_label = {"rand": "Unif.", "var": "IG-GP", "ei": "EI", "epd": "EPD"}[label]
    return clean_label


def save_video(rgbs, filename, fps=20.0):
    """
    Writes out a series of RGB arrays as a video with a specified framerate.
    Args:
        ims (list): numpy arrays if the same size
        filename (str): name of the output file (should end on .mp4)
        fps (float): frames per second
    """
    if filename[-4:] == ".gif":
        gif_output = True
        filename_gif = filename
        filename = filename_gif + ".mp4"
    else:
        gif_output = False

    if filename[-4:] != ".mp4":
        print("Warning: filename should end on .mp4, not '{}'".format(filename))

    # write .mp4
    ims = [cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) for rgb in rgbs]
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    (height, width, _) = ims[0].shape
    writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    for im in ims:
        writer.write(im)
    writer.release()

    if gif_output:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                filename,
                filename_gif,
            ]
        )
        os.remove(filename)


def record_gym_video(env, policy, filename):
    """
    Rollout a policy in a gym environment and record a video of its performance.
    Args:
        env (gym.Env): an environment that supports rgb_array rendering
        policy: a policy object that implements a get_action method
    """
    obs = env.reset()
    rgb = env.render("rgb_array")
    rgb = np.array(rgb, dtype=np.uint8)
    images = [rgb]
    done = False
    while not done:  # np all for vectorized envs
        a = policy.get_action(obs)
        obs, rew, done, info = env.step(a)
        rgb = env.render("rgb_array")
        rgb = np.array(rgb, dtype=np.uint8)
        images.append(rgb)
    save_video(images, filename)


def get_2d_direction_points(direction: np.ndarray, scale_max=2, scale_min=-2):
    direction = direction.squeeze()
    direction = direction / np.linalg.norm(direction)
    point1 = scale_max * direction
    point1 = np.array([-point1[1], point1[0]])
    point2 = scale_min * direction
    point2 = np.array([-point2[1], point2[0]])

    return point1, point2
