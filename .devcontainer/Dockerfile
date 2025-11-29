FROM python:3.11-slim

# Install system dependencies (including FCL + OctoMap for python-fcl)
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    libfcl-dev \
    liboctomap-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install \
        "numpy>=1.24.0" \
        "scipy>=1.10.0" \
        "matplotlib>=3.7.0" \
        "shapely>=1.8.0" \
        "python-fcl>=0.7.0" \
        "pytest>=7.3.0" \
        "black>=23.3.0" \
        "pylint>=2.17.0"

# Default command
CMD ["/bin/bash"]