import numpy as np

from src.policies.base_policy import BasePolicy


class FixedPolicy(BasePolicy):
    def __init__(self, policy: np.ndarray):
        self.matrix = np.copy(policy)

    def get_action(self, state, deterministic=True):
        t = int(state[-1])
        return self.matrix[t]

    def __eq__(self, other):
        return np.all(self.matrix == other.matrix)
