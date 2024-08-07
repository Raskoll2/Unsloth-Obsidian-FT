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
  pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
  pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
else
  echo "Installing Unsloth..."
  pip install unsloth
  pip install peft accelerate bitsandbytes xformers trl
fi

# Run the Python script
python main.py
