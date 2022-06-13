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
PRIOR_VARIANCE_SCALE = 1
ALGORITHM = "bounded_coordinate_hessian"
SIMULATION_STEPS = 1000
NUM_CANDIDATE_POLICIES = 25
NUM_QUERY = 200  # number of states, the number of queries will be n*(n-1)/4
CANDIDATE_POLICY_UPDATE_RATE = 50
QUERY_LOGGING_RATE = 5
IDRL = True
USE_ROLLOUTS = False
