DIMENSIONALITY = 8
THETA_NORM = 1
X_MIN = [
    -0.7,  # x distance
    -0.2,  # y distance
    -np.pi,  # heading angle
    -1,  # velocity
    0,  # distance to the car
]
X_MAX = [
    0.7,  # x distance
    0.2,  # y distance
    np.pi,  # heading angle
    1,  # velocity
    1.456,  # distance to the car
]
PRIOR_VARIANCE_SCALE = 1
ALGORITHM = "bounded_hessian"
SIMULATION_STEPS = 3000
