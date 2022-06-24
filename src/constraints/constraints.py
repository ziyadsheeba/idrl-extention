from abc import ABC, abstractmethod
from typing import List, Tuple

import cvxpy as cp
import numpy as np


class Constraint(ABC):
    @abstractmethod
    def get_cvxpy_constraint(self) -> Tuple[cp.Variable, List]:
        pass


class AffineConstraint(Constraint):
    def __init__(self, A: np.ndarray, b: np.ndarray):
        """Defines a constraint of the form Ax <= b.

        Args:
            A (np.ndarray)
            b (np.ndarray)
        """
        assert A.shape[1] == b.shape[0], "Dimensions of A and b don't match"
        self.A = A
        self.b = b
        self.dim = A.shape[1]

    def get_cvxpy_constraint(self) -> Tuple[cp.Variable, List]:
        x = cp.Variable((self.dim, 1))
        constraints = [A @ x <= b]
        return constraints, x


class EllipticalConstraint(Constraint):
    def __init__(self, A: np.ndarray, q: np.ndarray, b: float):
        """Defines a constraint of the form x.T@A@x + q@x <= b.

        Args:
            A (np.ndarray): Potential matrix.
            q (np.ndarray): Inner product vector.
            b (float): Level-set.
        """
        assert A.shape[0] == A.shape[1]
        self.A = A
        self.b = b
        self.q = q
        self.dim = A.shape[0]

    def get_cvxpy_constraint(self) -> Tuple[cp.Variable, List]:
        x = cp.Variable((self.dim, 1))
        constraints = [cp.quad_form(x, self.A) + self.q @ x <= self.b]
        return constraints, x


class SphericalConstraint(Constraint):
    def __init__(self, b: float, dim: int):
        """Defines a constraint of the form x.T@x <= b.

        Args:
            b (float): L2 Level-set.
            dim (int): Dimensionality of the variable.
        """
        self.b = b
        self.dim = dim

    def get_cvxpy_constraint(self) -> Tuple[cp.Variable, List]:
        x = cp.Variable((self.dim, 1))
        constraints = [cp.quad_form(x, np.eye(self.dim)) <= self.b]
        return constraints, x


class BoxConstraint(Constraint):
    def __init__(self, dim: int, lower: float, upper: float):
        """Defines a constraint of the form x<= upper, x>= lower, applied coordinate wise.

        Args:
            dim (int): The dimension of the vector.
            lower (float): Lower bound.
            upper (float): Upper bound.
        """
        self.lower = lower
        self.upper = upper
        self.dim = dim

    def get_cvxpy_constraint(self) -> Tuple[cp.Variable, List]:
        x = cp.Variable((self.dim, 1))
        constraints = [x <= self.upper, x >= self.lower]
        return constraints, x
