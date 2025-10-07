"""pytorch_training.thesis package

This package contains modules for XRF spectral analysis, data preprocessing,
model architectures, training scripts, and utilities.
Version: 0.1.0
"""

# configs
from .configs.config_reader import load_config

# wrappers and utilities
from .utils.wrappers import deprecated, LabelMapper
from .utils import tools, mappings, callbacks, plotting

# models
from .models.lightning.xrf_model import XRFClassifier
from .models.architectures.cnn import (
    Baseline1DCNN,
    SEAttention1DCNN,
    MHAAttention1DCNN,
    HybridAttention1DCNN,
    CNNAttentionBiLSTM,
    CNNLSTMResNet1D,
    LSTM1DCNN,
)

# dataset loaders
from .preprocessing.dataset_loader import SimpleSpectraLDM, SpectraData
from .preprocessing.transforms import (
    # Concat2TensorSoft,
    Concat2TensorSoftRealData,
    BaselineCorrection,
    Smoothing,
    Shift,
    Gain,
    LogarithmicTransform,
    AddNoise,
    AddPileup,
    Crop,
    NormIT,
    NormSqrt,
    SamplePreparation,
)

__all__ = [
    "load_config",
    "deprecated",
    "LabelMapper",
    "tools",
    "mappings",
    "callbacks",
    "plotting",
    "XRFClassifier",
    "Baseline1DCNN",
    "SEAttention1DCNN",
    "MHAAttention1DCNN",
    "HybridAttention1DCNN",
    "CNNAttentionBiLSTM",
    "CNNLSTMResNet1D",
    "LSTM1DCNN",
    "SimpleSpectraLDM",
    "RealSpectraData",
    "SpectraData",
    "Concat2TensorSoft",
    "Concat2TensorSoftRealData",
    "BaselineCorrection",
    "Smoothing",
    "Shift",
    "Gain",
    "LogarithmicTransform",
    "SetDefaultPileup",
    "AddPileup",
    "Crop",
    "AddNoise",
    "NormIT",
    "NormSqrt",
    "Compound",
    "PSF",
    "SpectrumHandler",
    "SamplePreparation",
]


__version__ = "0.1.0"
