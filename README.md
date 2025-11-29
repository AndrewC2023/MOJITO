# MOJITO
Maneuver-Oriented dynamics-aware Joint Iterative Trajectory Optimization

## Dependencies

This project uses [gncpy](https://github.com/drjdlarson/gncpy) as a git submodule.

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

