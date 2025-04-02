#!/bin/bash

# Ensure script fails on error
set -e

echo "Installing required Python packages..."
pip3 install --upgrade pip

# Install core packages for ML and data processing
pip3 install \
    numpy \
    scikit-learn \
    matplotlib \
    seaborn \
    tqdm \
    xgboost



echo "All requirements installed successfully."
