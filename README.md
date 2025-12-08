# MOJITO
**M**aneuver-**O**riented dynamics-aware **J**oint **I**terative **T**rajectory **O**ptimization

A sampling-based Model Predictive Control (MPC) framework for robotic systems with collision avoidance. MOJITO combines population-based optimization (Cross-Entropy Method) with dynamics-aware trajectory generation and FCL-based 3D collision detection.

## Key Features

- **Sampling-Based MPC**: Uses Cross-Entropy Method (CEM) for trajectory optimization
- **3D Collision Avoidance**: FCL (Flexible Collision Library) integration with soft proximity gradients
- **Time-Normalized Costs**: Physics timestep-invariant cost functions to model Integration of cost in the optimal control problem
- **Multiple Input Functions**: Supports spline interpolation and piecewise-constant control parameterization
- **Flexible Dynamics**: Uses the University of Alabama's LAGER repository: [gncpy](https://github.com/drjdlarson/gncpy) for dynamics modeling

## Architecture

```
src/
    ConfigurationSpace/   # 3D configuration space with FCL collision detection
    Controls/             # NACMPC controller and input function classes
    Optimizers/           # Cross-Entropy Method and optimizer base classes
    Utils/                # Geometry utilities (quaternions, rotations)
    Vehicles/             # Vehicle abstraction layer for dynamics and collision geometry
```

## Development Setup

### Using Dev Container (Recommended)

This project uses a dev container for consistent development environments.

#### With VS Code

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop)
2. Install the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) in VS Code
3. Open this folder in VS Code
4. When prompted, click "Reopen in Container" or navigate through the command pallete
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
## Contributing

When adding new features  
1. Checkout a new brach with the name of your feature
2. Develop and test it
3. Create a pull/merge request with the feature and test(s) for review

*Note: When implementing new dynamics or cost functions:
1. Ensure all running costs multiply by `dt` for time normalization
2. Test with multiple `physics_dt` values to verify cost consistency
3. Document coordinate frame conventions (NED vs ENU and body frame convensions)


## Future Work
This project in general needs a lot of work and research to find better solutions and apply good well known current
solutions. This framework hopes to be a place where MPC and search algorithm reseach can be done. It is likely that
at some point in the future the work here will be movesd to gncpy at which point it will be noted that this repository is stale.
