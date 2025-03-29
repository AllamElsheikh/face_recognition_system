#!/bin/bash
# Install system dependencies
apt-get install -y libomp-dev

# Install Python dependencies from requirements.txt
pip install -r requirments.txt
