from numpy import pi

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
PRIOR_VARIANCE_SCALE = 0.01
ALGORITHM = "current_map_hessian"
SIMULATION_STEPS = 1000
NUM_CANDIDATE_POLICIES = 10
NUM_QUERY = 100  # number of states, the number of queries will be n*(n-1)/4
TRAJECTORY_QUERY = True  # whether to use trajectory queries or not
CANDIDATE_POLICY_UPDATE_RATE = 10
QUERY_LOGGING_RATE = 1
IDRL = True
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]
N_PROCESSES = 8
