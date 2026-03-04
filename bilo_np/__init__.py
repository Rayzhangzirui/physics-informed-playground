"""BILO: Bilevel Optimization for Physics-Informed Neural Networks.

PDE: u' = a*u
Trial solution: u(t,a;W) = 1 + t*N(t,a;W)
"""

from model import BILOModel, BILOModelTorch

__all__ = ["BILOModel", "BILOModelTorch"]
