from typing import Union

import gym
import numpy as np
from warning import warn

from src.policies.base_policy import BasePolicy


class EpsGreedyPolicy(BasePolicy):
    def __init__(self, greedy_policy: BasePolicy, eps: float, action_space: gym.Space):
        """_summary_

        Args:
            greedy_policy (BasePolicy): A base policy.
            eps (float): Epsilon exploration parameter. Should be in (0,1).
            action_space (gym.Space): The action space of the agent.
        """
        assert eps >= 0 and eps <= 1
        if eps == 0:
            warn("eps is set to 0, no exploration will be performed")
        elif eps == 1:
            warn("eps is set to 1, only exploration will be performed")

        self.greedy = greedy_policy
        self.eps = eps
        self.action_space = action_space

    def get_action(self, obs: Union[int, np.ndarray], deterministic: bool = False):
        if deterministic or np.random.random() > self.eps:
            return self.greedy.get_action(obs, deterministic=True)
        else:
            return self.action_space.sample()
