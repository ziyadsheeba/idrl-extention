import gym
import numpy as np

from src.policies.base_policy import BasePolicy


class TabularPolicy(BasePolicy):
    def __init__(self, policy: np.ndarray):
        self.matrix = np.copy(policy)

    def get_action(self, state: int, deterministic: bool = True) -> int:
        """
        Args:
            state (int): The current state.
            deterministic (bool, optional): Wether to sample or pick the maximum. Defaults to True.

        Returns:
            int: Action
        """
        if deterministic:
            return np.argmax(self.matrix[state, :])
        else:
            return np.random.choice(
                range(self.matrix.shape[1]), p=self.matrix[state, :]
            )

    def evaluate(self, env: gym.Env, N: int = 1, rollout=False):
        """
        Args:
            env (gym.Env): A gym-like environment. The environment must implement the function 'evaluate_policy.'
            N (int, optional): Number of rollouts. Defaults to 1.
            rollout (bool, optional): Wether to perform a rollout or not. Defaults to False.

        Returns:
            _type_: _description_
        """
        assert env.observation_type == "state"
        if rollout:
            return super().evaluate(env, N)
        else:
            return env.evaluate_policy(self)

    def __eq__(self, other):
        return np.all(self.matrix == other.matrix)
