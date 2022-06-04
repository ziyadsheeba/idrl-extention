import numpy as np

from src.policies.base_policy import BasePolicy


class CombinedPolicy(BasePolicy):
    def __init__(self, policies, p=None):
        self.policies = policies
        for policy in self.policies:
            assert issubclass(policy.__class__, BasePolicy)
        if p is None:
            n = len(self.policies)
            p = np.ones(n) / n
        self.p = p

    def get_action(self, obs, deterministic=True):
        policy_idx = np.random.choice(np.arange(len(self.policies)), p=self.p)
        policy = self.policies[policy_idx]
        return policy.get_action(obs, deterministic=deterministic)
