import numpy as np

from src.policies.base_policy import BasePolicy


class GaussianNoisePolicy(BasePolicy):
    def __init__(self, policy: BasePolicy, sigma: float):
        self.policy = policy
        self.sigma = sigma

    def get_action(self, obs, deterministic=False):
        action = self.policy.get_action(obs, deterministic=deterministic)
        action += np.random.normal(loc=0, scale=self.sigma, size=action.shape)
        return action
