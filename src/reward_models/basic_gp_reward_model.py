import os
import pickle
import time
from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple, Union

import gym
import numpy as np
import scipy.sparse as sp

from src.envs.reward_model_mean_wrapper import RewardModelMeanWrapper
from src.envs.reward_model_sample_wrapper import RewardModelSampleWrapper
from src.envs.tabular_mdp import TabularMDP
from src.policies.base_policy import BasePolicy
from src.policies.combined_policy import CombinedPolicy
from src.policies.eps_greedy_policy import EpsGreedyPolicy
from src.policies.gaussian_noise_policy import GaussianNoisePolicy
from src.policies.linear_policy import LinearPolicy
from src.reward_models.gaussian_process_linear import LinearObservationGP
from src.reward_models.kernels import LinearKernel
from src.reward_models.query import (
    ComparisonQueryLinear,
    LinearQuery,
    PointQuery,
    QueryBase,
    StateComparisonQueryLinear,
    StateComparisonQueryNonlinear,
    StateQuery,
    TrajectoryQuery,
)
from src.solvers.argmax_solver import ArgmaxSolver
from src.solvers.base_solver import BaseSolver
from src.solvers.lp_solver import LPSolver
from src.utils import (
    get_hash,
    mean_jaccard_distance,
    np_to_tuple,
    subsample_sequence,
    timing,
)
