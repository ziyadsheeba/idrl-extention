import os
import pickle
from multiprocessing import Pool

import numpy as np
import tqdm
from joblib import Parallel, delayed
from tqdm import tqdm

from src.agents.gp_agent import GPAgent
from src.agents.linear_agent import LinearAgent
from src.constants import (
    DRIVER_PRECOMPUTED_POLICIES_PATH,
    DRIVER_STATES_TESTSET_PATH,
    DRIVER_TRAJECTORIES_TESTSET_PATH,
)
from src.envs.driver import get_driver_target_velocity
from src.reward_models.kernels import RBFKernel
from src.reward_models.logistic_reward_models import (
    GPLogisticRewardModel,
    LinearLogisticRewardModel,
)

N_SAMPLES = 1000
NUM_QUERY = 1000
SEED = 0
N_JOBS = 8


def get_testset_raw_states():
    testset = {}
    from src.nonlinear.driver_config import DIMENSIONALITY

    env = get_driver_target_velocity()

    reward_model = GPLogisticRewardModel(
        dim=DIMENSIONALITY,
        kernel=RBFKernel(dim=DIMENSIONALITY),
        trajectory=False,
    )

    agent = GPAgent(
        query_expert=env.get_comparison_from_full_states,
        get_representation=env.get_full_state,
        get_render_representation=env.get_render_state,
        get_optimal_policy_from_reward_function=env.get_optimal_policy_from_reward_function,
        env_step=env.step,
        env_reset=env.reset,
        precomputed_policy_path=DRIVER_PRECOMPUTED_POLICIES_PATH / "policies.pkl",
        reward_model=reward_model,
        num_candidate_policies=None,
        idrl=False,
        candidate_policy_update_rate=None,
        representation_space_dim=DIMENSIONALITY,
        use_trajectories=False,
        num_query=NUM_QUERY,
        n_jobs=1,
    )
    global sampler

    def sampler(seed):
        np.random.seed(seed)  # Necessary, otherwise duplicates  occur
        query, label, argmax = agent.optimize_query(
            algorithm="random", n_jobs=N_JOBS, feedback_mode="soft"
        )
        return ((query[0].tobytes(), query[1].tobytes()), label), query[0].shape

    testset = []
    pool = Pool(processes=N_JOBS)
    for test_point, query_shape in tqdm(
        pool.imap_unordered(sampler, range(N_SAMPLES)),
        total=N_SAMPLES,
    ):
        testset.append(test_point)
    testset = dict(testset)
    return testset, query_shape


def get_testset_trajectory_states():
    testset = {}
    from src.nonlinear.driver_config import DIMENSIONALITY

    env = get_driver_target_velocity()

    reward_model = GPLogisticRewardModel(
        dim=DIMENSIONALITY,
        kernel=RBFKernel(dim=DIMENSIONALITY),
        trajectory=True,
    )

    agent = GPAgent(
        query_expert=env.get_comparison_from_full_states,
        get_representation=env.get_full_state,
        get_render_representation=env.get_render_state,
        get_optimal_policy_from_reward_function=env.get_optimal_policy_from_reward_function,
        env_step=env.step,
        env_reset=env.reset,
        precomputed_policy_path=DRIVER_PRECOMPUTED_POLICIES_PATH / "policies.pkl",
        reward_model=reward_model,
        num_candidate_policies=None,
        idrl=False,
        candidate_policy_update_rate=None,
        representation_space_dim=DIMENSIONALITY,
        use_trajectories=True,
        num_query=NUM_QUERY,
        n_jobs=1,
    )
    global sampler

    def sampler(seed):
        np.random.seed(seed)  # Necessary, otherwise duplicates  occur
        query, label, argmax = agent.optimize_query(
            algorithm="random", n_jobs=N_JOBS, feedback_mode="soft"
        )
        return ((query[0].tobytes(), query[1].tobytes()), label), query[0].shape

    testset = []
    pool = Pool(processes=N_JOBS)
    for test_point, query_shape in tqdm(
        pool.imap_unordered(sampler, range(N_SAMPLES)),
        total=N_SAMPLES,
    ):
        testset.append(test_point)
    testset = dict(testset)
    return testset, query_shape


def main():

    # Create a testset with raw states
    testset, query_shape = get_testset_raw_states()
    os.makedirs(DRIVER_STATES_TESTSET_PATH, exist_ok=True)

    with open(f"{str(DRIVER_STATES_TESTSET_PATH)}/testset.pkl", "wb") as f:
        pickle.dump(testset, f)

    with open(f"{str(DRIVER_STATES_TESTSET_PATH)}/query_shape.pkl", "wb") as f:
        pickle.dump(query_shape, f)

    # Create a testset with the raw trajectories
    testset, query_shape = get_testset_trajectory_states()
    os.makedirs(DRIVER_TRAJECTORIES_TESTSET_PATH, exist_ok=True)

    with open(f"{str(DRIVER_TRAJECTORIES_TESTSET_PATH)}/testset.pkl", "wb") as f:
        pickle.dump(testset, f)

    with open(f"{str(DRIVER_TRAJECTORIES_TESTSET_PATH)}/query_shape.pkl", "wb") as f:
        pickle.dump(query_shape, f)


if __name__ == "__main__":
    # np.random.seed(SEED)
    main()
