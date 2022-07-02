import pickle
from multiprocessing import Pool

import numpy as np
import tqdm

from src.constants import DRIVER_PRECOMPUTED_POLICIES_PATH
from src.envs.driver import get_driver_target_velocity
from src.linear.driver_config import (
    DIMENSIONALITY,
    PRIOR_VARIANCE_SCALE,
    THETA_NORM,
    X_MAX,
    X_MIN,
)
from src.reward_models.logistic_reward_models import LinearLogisticRewardModel

N_POLICIES = 1000


def compute_optimal_policy(theta):
    env = get_driver_target_velocity()
    policy = env.get_optimal_policy(theta=theta)
    return policy


def main():

    # Initialize the reward model
    reward_model = LinearLogisticRewardModel(
        dim=DIMENSIONALITY,
        prior_variance=PRIOR_VARIANCE_SCALE * (THETA_NORM) ** 2 / 2,
        param_norm=THETA_NORM,
        x_min=X_MIN,
        x_max=X_MAX,
    )

    # sample from the prior distribution
    samples = reward_model.sample_current_approximate_distribution(n_samples=N_POLICIES)
    pool = Pool(processes=8)
    policies = []
    for policy in tqdm.tqdm(
        pool.imap_unordered(compute_optimal_policy, samples), total=len(samples)
    ):
        policies.append(policy)

    # save policies
    with open(f"{str(DRIVER_PRECOMPUTED_POLICIES_PATH)}/policies.pkl", "wb") as f:
        pickle.dump(policies, f)


if __name__ == "__main__":
    main()
