from numpy import pi

DIMENSIONALITY = 8
ALGORITHM = "random"
SIMULATION_STEPS = 10000
NUM_CANDIDATE_POLICIES = 8
NUM_QUERY = 200  # number of states, the number of queries will be n*(n-1)/4
TRAJECTORY_QUERY = True  # whether to use trajectory queries or not
CANDIDATE_POLICY_UPDATE_RATE = 1
QUERY_LOGGING_RATE = 1
IDRL = False
SEEDS = [0]
N_JOBS = 1
