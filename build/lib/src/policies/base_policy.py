import datetime
import os
import pickle
from abc import ABC, abstractmethod

import gym
import numpy as np


class BasePolicy(ABC):
    @abstractmethod
    def get_action(self, obs: np.ndarray, deterministic: bool = True):
        raise NotImplementedError()

    def evaluate(self, env: gym.Env, N: int = 10, rollout: bool = True) -> float:
        """policy evaluation method

        Args:
            env (gym.Env): The environment to interact in.
            N (int, optional): Number of rollouts. Defaults to 10.
            rollout (bool, optional): Wether to perform a rollout or not. Defaults to True.

        Returns:
            float: Average reward over rollouts
        """
        if not rollout:
            print("Warning: Rolling out policy despite rollout=False")
        res = 0
        for _ in range(N):
            obs = env.reset()
            done = False
            while not done:
                a = self.get_action(obs)
                obs, reward, done, _ = env.step(a)
                res += reward
        return res / N
