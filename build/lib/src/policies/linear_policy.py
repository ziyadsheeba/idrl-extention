import pickle

import numpy as np

from src.policies.base_policy import BasePolicy


class LinearPolicy(BasePolicy):
    def __init__(self, w, obs_mean=None, obs_std=None, env=None):
        self.w = w
        self.obs_mean = obs_mean
        self.obs_std = obs_std
        if env is not None:
            self.alow = env.action_space.low
            self.ahigh = env.action_space.high
        else:
            self.alow = -np.inf
            self.ahigh = np.inf

    def normalize(self, obs):
        if self.obs_mean is not None and self.obs_std is not None:
            return (obs - self.obs_mean) / self.obs_std
        else:
            return obs

    def get_action(self, obs, deterministic=True):
        obs = self.normalize(obs)
        a = np.dot(self.w, obs)
        a = np.clip(a, self.alow, self.ahigh)
        return a

    def save(self, path):
        policy_dict = {
            "w": list(self.w),
            "mean": list(self.obs_mean),
            "std": list(self.obs_std),
        }
        with open(path, "wb") as f:
            pickle.dump(policy_dict, f)

    @classmethod
    def load(cls, path, env=None):
        with open(path, "rb") as f:
            policy_dict = pickle.load(f)
        policy = cls(
            policy_dict["w"],
            obs_mean=policy_dict["mean"],
            obs_std=policy_dict["std"],
            env=env,
        )
        return policy
