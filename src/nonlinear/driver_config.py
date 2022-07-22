from numpy import pi

from src.constants import DRIVER_STATES_TESTSET_PATH, DRIVER_TRAJECTORIES_TESTSET_PATH

DIMENSIONALITY = 8
ALGORITHM = "predicted_variance"
SIMULATION_STEPS = 200
NUM_CANDIDATE_POLICIES = 8
NUM_QUERY = 400  # number of states, the number of queries will be n*(n-1)/4
TRAJECTORY_QUERY = True  # whether to use trajectory queries or just states
CANDIDATE_POLICY_UPDATE_RATE = 1
QUERY_LOGGING_RATE = 1
IDRL = False
SEEDS = [0]
N_JOBS = 8
KERNEL_PARAMS = {
    "dim": DIMENSIONALITY,
    "lengthscale": 1,
    "obs_var": 1e-8,
    "variance": 1,
    "constant": 0,
}
if TRAJECTORY_QUERY:
    TESTSET_PATH = DRIVER_TRAJECTORIES_TESTSET_PATH
else:
    TESTSET_PATH = DRIVER_STATES_TESTSET_PATH
