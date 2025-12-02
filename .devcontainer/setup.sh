#!/bin/bash
# Setup script for MOJITO development environment
# This script is run automatically by devcontainer or can be run manually

set -e  # Exit on error

echo "=== MOJITO Development Environment Setup ==="

# Configure git settings
echo "Configuring git..."
git config --local core.autocrlf input
git config --local core.eol lf

# Initialize and update submodules
echo "Initializing submodules..."
git submodule update --init --recursive
git submodule set-branch --branch OmnicoptorModelAndControl dependencies/gncpy
git submodule update --remote dependencies/gncpy

# Install gncpy in editable mode
echo "Installing gncpy..."

cd dependencies/gncpy
pip install -e .
cd ../..

echo "=== Setup complete! ==="
