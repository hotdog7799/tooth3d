"""
Regularizers package for ADMM-based optimization.

This package provides modular regularizers that can be easily swapped
in and out of ADMM solvers for different reconstruction tasks.
"""

from .base_regularizer import BaseRegularizer
from .tv3d_regularizer import TV3DRegularizer
from .l1_regularizer import L1Regularizer
from .center_weighted_regularizer import CenterWeightedRegularizer
from .anisotropic_diffusion_regularizer import AnisotropicDiffusionRegularizer

__all__ = ['BaseRegularizer', 'TV3DRegularizer', 'L1Regularizer', 'CenterWeightedRegularizer', 'AnisotropicDiffusionRegularizer'] 