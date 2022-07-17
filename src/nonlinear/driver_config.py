from numpy import pi

DIMENSIONALITY = 8
ALGORITHM = "random"
SIMULATION_STEPS = 10000
NUM_CANDIDATE_POLICIES = 8
NUM_QUERY = 200  # number of states, the number of queries will be n*(n-1)/4
TRAJECTORY_QUERY = False  # whether to use trajectory queries or just states
CANDIDATE_POLICY_UPDATE_RATE = 1
QUERY_LOGGING_RATE = 1
IDRL = False
SEEDS = [0]
N_JOBS = 8
KERNEL_PARAMS = {"dim": DIMENSIONALITY, "lengthscale": 1, "obs_var": 1e-8}
