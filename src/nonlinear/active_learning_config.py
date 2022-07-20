"""Experiments Config 
"""
ALGORITHM = "random"
DIMENSIONALITY = 1
X_MIN = -3
X_MAX = 3
N_SAMPLES = 50
PLOT = False
SEED = 10
SIMULATION_STEPS = 1000
KERNEL_PARAMS = {
    "dim": DIMENSIONALITY,
    "lengthscale": 1,
    "obs_var": 1e-8,
    "variance": 10,
    "constant": 0,
}
