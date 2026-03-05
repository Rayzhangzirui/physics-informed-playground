"""BILO: Bilevel Optimization for Physics-Informed Neural Networks.

PDE: u' = a*u or u' = a*u*(1-u) (logistic).
Trial solution: u = 1 + t*N (exponential) or u = u0 + t*N (logistic).
"""

from .model import BILOModel, PINNModel, BILOModelTorch, PINNModelTorch, logistic_solution

__all__ = ["BILOModel", "PINNModel", "BILOModelTorch", "PINNModelTorch", "logistic_solution"]
