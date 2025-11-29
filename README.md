# MOJITO
Maneuver-Oriented dynamics-aware Joint Iterative Trajectory Optimization

## Development Setup

### Using Dev Container (Recommended)

This project uses a dev container for consistent development environments:

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop)
2. Install the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) in VS Code
3. Open this folder in VS Code
4. When prompted, click "Reopen in Container" (or use Command Palette: `Dev Containers: Reopen in Container`)
5. The container will build and install all dependencies including PyTorch

The dev container includes:
- Python 3.11
- PyTorch (CPU version by default)
- All project dependencies from `requirements.txt`
- Python extensions for VS Code

### Local Setup (Alternative)

There is currently not clear support on local setup. Refer to the dockerfile in the devcontainer folder for the necessary dependencies.

**Note:** For GPU support with PyTorch, modify the dockerfile to enable CUDA or parallelization support

## Dependencies

This project uses [gncpy](https://github.com/drjdlarson/gncpy) as a git submodule.
This project also uses PyTorch3D for 3D operations and rendering. Install one of the following depending on your PyTorch/CUDA setup:

- Install via pip (use wheel matching your PyTorch/CUDA):
```bash
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/
```

- Or build/install from source:
```bash
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
pip install -e .
```

See the PyTorch3D installation guide for exact wheel selection: https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md

If using the devcontainer, add the appropriate installation step to the Dockerfile or requirements.txt.

### Cloning the Repository

For new clones, use:
```bash
git clone --recurse-submodules https://github.com/AndrewC2023/MOJITO.git
```

Or if you've already cloned without submodules:
```bash
git submodule update --init --recursive
```

### Updating Submodules

To update the gncpy dependency to the latest version:
```bash
git submodule update --remote dependencies/gncpy
```

