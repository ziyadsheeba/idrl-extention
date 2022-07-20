from numpy import pi

from src.constants import DRIVER_STATES_TESTSET_PATH, DRIVER_TRAJECTORIES_TESTSET_PATH

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
PRIOR_VARIANCE_SCALE = 1
ALGORITHM = "current_map_hessian"
SIMULATION_STEPS = 200
NUM_CANDIDATE_POLICIES = 8
NUM_QUERY = 1000  # number of states, the number of queries will be n*(n-1)/4
TRAJECTORY_QUERY = True  # whether to use trajectory queries or not
CANDIDATE_POLICY_UPDATE_RATE = 1
QUERY_LOGGING_RATE = 1
IDRL = False
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]
N_JOBS = 8
if TRAJECTORY_QUERY:
    TESTSET_PATH = DRIVER_TRAJECTORIES_TESTSET_PATH
else:
    TESTSET_PATH = DRIVER_STATES_TESTSET_PATH
