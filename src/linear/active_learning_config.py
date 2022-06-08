"""Experiments Config 
"""
ALGORITHMS = [
    "map_confidence",
    "map_hessian",
    "current_map_hessian",
    "bounded_coordinate_hessian",
    "optimal_hessian",
    "random",
    "bounded_hessian",
]
DIMENSIONALITY = [2]
THETA_NORM = 1
EXPERT_SCALE = 40
X_MIN = -5
X_MAX = 5
GRID_RES = 70j
PRIOR_VARIANCE_SCALE = 1
PLOT = True
SEEDS = [0, 1, 2]
SIMULATION_STEPS = 10000
