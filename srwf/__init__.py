"""Scalable Random Wavelet Features."""

from .blr import BayesianLinearRegression
from .features import RandomFourierFeatures, RandomWaveletFeatures
from .model import RFFGP, RWFGP, MixtureGP, Prediction
from .wavelets import get_wavelet

__version__ = "1.1.0"

__all__ = [
    "BayesianLinearRegression",
    "MixtureGP",
    "Prediction",
    "RFFGP",
    "RWFGP",
    "RandomFourierFeatures",
    "RandomWaveletFeatures",
    "get_wavelet",
]
