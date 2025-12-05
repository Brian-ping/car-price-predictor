#!/bin/bash
# Force Python 3.9
export PYTHON_VERSION=3.9.0

# Install pip
python -m pip install --upgrade pip

# Install requirements
pip install --only-binary=:all: scikit-learn
pip install -r requirements.txt
