from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


class Kernel(ABC):
    @abstractmethod
    def eval(self):
        pass


class LinearKernel(Kernel):
    def __init__(self, dim: int, obs_var: float = 1e-8):
        """An implementation of a linear kernel function

        Args:
            input_dim (int): The covariates dimension.
            variances (List[float], optional): Coordinates scale for the
                inner product. Defaults to None.
        """
        self.dim = dim
        self.obs_var = obs_var

    def eval(self, X_1: np.ndarray, X_2: np.ndarray) -> np.ndarray:
        """Kernel function.

        Args:
            x (np.ndarray)
            y (np.ndarray)

        Returns:
            float
        """
        assert (
            X_1.shape[1] == self.dim and X_2.shape[1] == self.dim
        ), "Input must be 2-d array"
        K = X_1 @ X_2.T
        if K.shape[0] == K.shape[1]:
            if K.shape[0] == 1:
                K = K.item() + self.obs_var
            else:
                K += np.eye(K.shape[0]) * self.obs_var
        return K


class RBFKernel(Kernel):
    """
    Radial basis function (RBF) kernel. The kernel can handle states and trajectories. The trajectory inputs
        should be of the shape (n_trajectories, dimension, len_trajectory)
    Attributes:
    -------------
    variance: RBF variance (sigma)
    lengthscale: RBF lengthscale (l)
    obs_var: Observation variance
    """

    def __init__(
        self,
        dim: int,
        variance: float = 1,
        lengthscale: float = 0.5,
        obs_var: float = 1e-8,
        use_cache: bool = False,
    ):
        self.dim = dim
        self.variance = variance
        self.lengthscale = lengthscale
        self.obs_var = obs_var

    def eval(self, X_1: np.ndarray, X_2: np.ndarray) -> np.ndarray:
        """Kernel evaluation function.

        Args:
            X_1 (np.ndarray)
            X_2 (np.ndarray)

        Returns:
            np.ndarray: Kernel value.
        """
        assert (
            X_1.ndim <= 3 and X_2.ndim <= 3
        ), "Kernel Inputs can be maximum 3 dimensional"
        norm = 0
        norm += np.sum(X_1**2, axis=1)[:, None] + np.sum(X_2**2, axis=1)[None, :]
        if X_1.ndim == X_2.ndim == 3:
            norm -= 2 * np.einsum("ijn,kjn->ikn", X_1, X_2)
            K = self.variance**2 * np.exp(-0.5 * norm / self.lengthscale**2).sum(
                axis=-1
            )
            if K.shape[0] == K.shape[1]:
                if K.shape[0] == 1:
                    K = K.item() + self.obs_var
                else:
                    K += np.eye(K.shape[0]) * self.obs_var * X_1.shape[-1]
        elif X_1.ndim == X_2.ndim == 2:
            norm -= 2 * np.dot(X_1, X_2.T)
            K = self.variance**2 * np.exp(-0.5 * norm / self.lengthscale**2)
            if K.shape[0] == K.shape[1]:
                if K.shape[0] == 1:
                    K = K.item() + self.obs_var
                else:
                    K += np.eye(K.shape[0]) * self.obs_var
        elif X_1.ndim == 2 and X_2.ndim == 3:
            norm -= 2 * np.einsum("ij,kjn->ikn", X_1, X_2)
            K = self.variance**2 * np.exp(-0.5 * norm / self.lengthscale**2).sum(
                axis=-1
            )
        return K
