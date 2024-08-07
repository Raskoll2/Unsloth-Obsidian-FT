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

  # Install specific version of torch compatible with CUDA 12.0
  pip install torch==2.3.0+cu121 torchvision==0.15.1+cu121 torchaudio==2.0.2 --extra-index-url https://download.pytorch.org/whl/cu121

  # Install xformers with specific CUDA version compatibility
  pip install "xformers<0.0.27"

  # Install Unsloth and other dependencies
  pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
  pip install --no-deps "trl<0.9.0" peft accelerate bitsandbytes

else
  echo "Installing Unsloth..."
  
  # Standard installation for non-Colab environments
  pip install unsloth
  pip install peft accelerate bitsandbytes xformers trl
fi

# Run the Python script
python main.py
