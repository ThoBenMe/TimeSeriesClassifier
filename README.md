# Attention1DCNN

This repository contains the implementation of a 1D Convolutional Neural Network (CNN) architecture enhanced with attention mechanisms for spectral data classification. The model is designed for research and experimentation on spectroscopic datasets, focusing on improving feature extraction and interpretability through channel and spatial attention.

## Overview

The project investigates how attention modules can be integrated into traditional CNN pipelines to enhance performance on high-dimensional spectral signals. The model extends a standard 1D-CNN backbone with lightweight attention components inspired by Squeeze-and-Excitation (SE) and Self-Attention architectures.

Key goals:
- Improve class discrimination in complex, overlapping spectral features.
- Explore hybrid architectures combining CNNs with attention mechanisms.
- Evaluate model transferability from synthetic to real-world spectra.

## Features

- Configurable CNN backbone with multiple convolutional layers.
- Optional attention modules (SE, CBAM, Multi-Head Self-Attention).
- Flexible data loading from `.h5` and `.bcf` formats.
- Integrated training, validation, and logging using PyTorch Lightning.
- Hyperparameter optimization via Optuna.
- Visualization tools for spectra and learned feature maps.


## Getting Started

1. Install dependencies:
```
pip install -r requirements.txt
```
2. Adjust parameters in configs/config.yml.
3. Train the model:
```
python train_model.py --model <modelname> --mix --epochs 100
```


Developed for research on spectral learning and model transferability.
