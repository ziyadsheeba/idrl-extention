import numpy as np

from src.utils import sample_random_ball


class Policy:
    def __init__(self, state_support_size: int, state_space_dim: int):
        """_summary_

        Args:
            state_support_size (int): _description_
            state_space_dim (int): _description_
        """
        self.X = np.concatenate(
            [sample_random_ball(state_space_dim) for _ in range(state_support_size)]
        )
        self.visitation_frequencies = np.random.randint(
            low=1, high=1000, size=(state_support_size, 1)
        )
        self.visitation_frequencies = self.visitation_frequencies
        self.v = self.X.T @ self.visitation_frequencies
