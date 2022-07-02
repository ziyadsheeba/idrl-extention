import copy
import unittest

import numpy as np
from numpy import pi
from scipy import spatial

from src.agents.linear_agent import LinearAgent as Agent
from src.constants import DRIVER_PRECOMPUTED_POLICIES_PATH
from src.envs.driver import get_driver_target_velocity
from src.reward_models.logistic_reward_models import LinearLogisticRewardModel

DIMENSIONALITY = 8
THETA_NORM = 2
X_MIN = [
    -0.7,  # x distance, agent
    -0.2,  # y distance, agent
    -pi,  # heading angle, agent
    -1,  # velocity, agent
    -0.7,  # x distance, other
    -0.2,  # y distance, other
]
X_MAX = [
    0.7,  # x distance
    0.2,  # y distance
    pi,  # heading angle
    1,  # velocity
    0.7,  # x distance
    0.2,  # y distance
]
PRIOR_VARIANCE_SCALE = 10000
ALGORITHM = "current_map_hessian"
NUM_CANDIDATE_POLICIES = 10
NUM_QUERY = 100  # number of states, the number of queries will be n*(n-1)/4
TRAJECTORY_QUERY = True  # whether to use trajectory queries or not
CANDIDATE_POLICY_UPDATE_RATE = 5
QUERY_LOGGING_RATE = 1
IDRL = True
SEEDS = [0]
N_PROCESSES = 1


class TestLinearAgent(unittest.TestCase):
    def test_svf_computation(self):

        # Initialize environment
        env = get_driver_target_velocity()

        # Initialize the reward model
        reward_model = LinearLogisticRewardModel(
            dim=DIMENSIONALITY,
            prior_variance=PRIOR_VARIANCE_SCALE * (THETA_NORM) ** 2 / 2,
            param_norm=THETA_NORM,
            x_min=X_MIN,
            x_max=X_MAX,
        )
        # Initialize the agents
        agent = Agent(
            query_expert=env.get_comparison_from_feature_diff,
            state_to_features=env.get_query_features,
            estimate_state_visitation=env.estimate_state_visitation,
            get_optimal_policy=env.get_optimal_policy,
            get_query_from_policies=env.get_query_from_policies,
            precomputed_policy_path=DRIVER_PRECOMPUTED_POLICIES_PATH / "policies.pkl",
            reward_model=reward_model,
            num_candidate_policies=NUM_CANDIDATE_POLICIES,
            idrl=IDRL,
            candidate_policy_update_rate=CANDIDATE_POLICY_UPDATE_RATE,
            state_space_dim=DIMENSIONALITY,
            use_trajectories=TRAJECTORY_QUERY,
            num_query=NUM_QUERY,
        )
        optimal_policy = env.get_optimal_policy()
        svf_diff_mean, states = agent.estimate_pairwise_svf_mean(
            [optimal_policy, optimal_policy]
        )
        self.assertTrue((svf_diff_mean == 0).all())
