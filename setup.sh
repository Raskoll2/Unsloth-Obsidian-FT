#!/bin/bash

# Check if torch is already installed
if ! python -c "import torch" &> /dev/null; then
  echo "Torch not found. Installing..."
  pip install torch
else
  echo "Torch already installed."
fi

# Install required libraries
pip install transformers datasets dropbox

# Handle Google Colab specific setup
if [ "$1" == "--colab" ]; then
  echo "Running in Google Colab. Installing Unsloth with Colab support..."
  pip install "unsloth[colab] @ git+https://github.com/unslothai/unsloth.git" --quiet
else
  echo "Installing Unsloth..."
  pip install unsloth
fi

# Run the Python script
python main.py
