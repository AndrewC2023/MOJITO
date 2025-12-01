# MOJITO
Maneuver-Oriented dynamics-aware Joint Iterative Trajectory Optimization

## Development Setup

### Using Dev Container (Recommended)

This project uses a dev container for consistent development environments.

#### With VS Code

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop)
2. Install the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) in VS Code
3. Open this folder in VS Code
4. When prompted, click "Reopen in Container" (or use Command Palette: `Dev Containers: Reopen in Container`)
5. The container will build and install all dependencies including PyTorch

#### Without VS Code (Docker CLI)

You can also build and run the container directly using Docker:

```bash
# Build the container
docker build -t mojito-dev -f .devcontainer/Dockerfile .

# Run the container with interactive shell
docker run -it --rm -v $(pwd):/workspace mojito-dev
```

The dev container includes:
- Python 3.11
- Free Collision Library
- Python extensions for VS Code


### Local Development Setup (Alternative)

There is currently not clear support on local setup. Refer to the dockerfile in the devcontainer folder for the necessary dependencies.

## Dependencies

This project uses [gncpy](https://github.com/drjdlarson/gncpy) as a git submodule.

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

