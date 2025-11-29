FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install \
        "numpy>=1.24.0" \
        "scipy>=1.10.0" \
        "matplotlib>=3.7.0" \
        "shapely>=1.8.0" && \
    # CPU-only PyTorch stack, pinned to a version supported by PyTorch3D
    pip install \
        "torch==2.2.0" \
        "torchvision==0.17.0" \
        "torchaudio==2.2.0" \
        --index-url https://download.pytorch.org/whl/cpu && \
    # PyTorch3D dependencies (ioPath), plus tooling
    pip install \
        iopath \
        "pytest>=7.3.0" \
        "black>=23.3.0" \
        "pylint>=2.17.0" && \
    # Install PyTorch3D from the stable branch (builds CPU ops against torch 2.2)
    pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# Default command
CMD ["/bin/bash"]
