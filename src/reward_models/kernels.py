from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


class Kernel(ABC):
    @abstractmethod
    def eval(self):
        pass


class LinearKernel(Kernel):
    def __init__(self, input_dim: int, variances: List[float] = None):
        """An implementation of a linear kernel function

        Args:
            input_dim (int): The covariates dimension.
            variances (List[float], optional): Coordinates scale for the
                inner product. Defaults to None.
        """
        self.input_dim = input_dim
        if variances is None:
            variances = np.ones(input_dim)
        self.variances = variances

    def eval(self, x: np.ndarray, y: np.ndarray) -> float:
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
        K = np.zeros(shape=(X_1.shape[0], X_2.shape[0]))
        for i, x_1 in enumerate(X_1):
            for j, x_2 in enumerate(X_2):
                key = (x_1.tostring(), x_2.tostring())
                if key in self.k_cache:
                    K[i, j] = self.k_cache[key]
                else:
                    k = np.matmul(x_1.T, np.matmul(np.diag(self.variances), x_2))
                    self.k_cache[key] = k
                    K[i, j] = k
        if K.shape[0] == K.shape[1] == 0:
            K = K.item()
        return K


class RBFKernel(Kernel):
    """
    Radial basis function (RBF) kernel that allows to specify a custom distance.
    Attributes:
    -------------
    variance: RBF variance (sigma)
    lengthscale: RBF lengthscale (l)
    distance: Distance function that takes two points and returns a float
    k_cache: used to cache the covariances that have already been calculated
    """

    def __init__(
        self,
        dim: int,
        variance: float = 1,
        lengthscale: float = 1,
        distance: Callable[[np.ndarray, np.ndarray], float] = lambda x, y: np.sum(
            np.sqrt(np.square(x - y).sum())
        ),
    ):
        self.dim = dim
        self.variance = variance
        self.lengthscale = lengthscale
        self.distance = distance
        self.k_cache: Dict[Tuple[str, str], float] = {}

    def eval(self, X_1: np.ndarray, X_2: np.ndarray) -> np.ndarray:
        """Kernel evaluation function.

        Args:
            X_1 (np.ndarray)
            X_2 (np.ndarray)

        Returns:
            np.ndarray: Kernel value.
        """
        assert (
            X_1.shape[1] == self.dim and X_2.shape[1] == self.dim
        ), "Input must be 2-d array"
        K = np.zeros(shape=(X_1.shape[0], X_2.shape[0]))
        for i, x_1 in enumerate(X_1):
            for j, x_2 in enumerate(X_2):
                key = (x_1.tobytes(), x_2.tobytes())
                if key in self.k_cache:
                    K[i, j] = self.k_cache[key]
                else:
                    r = self.distance(x_1, x_2) / self.lengthscale
                    k = self.variance**2 * np.exp(-0.5 * r**2).item()
                    self.k_cache[key] = k
                    K[i, j] = k
        if K.shape[0] == K.shape[1] == 0:
            K = K.item()
        return K
