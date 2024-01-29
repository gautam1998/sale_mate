#!/bin/bash

# Print the current working directory
echo "Current working directory:"
pwd

# Change the directory
cd ../../

# Clone the Git repository
echo "Cloning the repository..."
git clone https://github.com/jaywalnut310/vits.git

# Check Python version
echo "Python version:"
python --version

# Navigate to the 'vits' directory
cd vits/

# Navigate to the 'monotonic_align' directory
cd monotonic_align/

# Create a directory 'monotonic_align'
mkdir monotonic_align

# Build the extension using setup.py
echo "Building the extension..."
python3 setup.py build_ext --inplace

# Move back to the parent directory
cd ../

# Print the current working directory again
echo "Current working directory:"
pwd
