"""Test script demonstrating Vehicle class usage with gncpy dynamics."""
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))
sys.path.append(str(Path(__file__).parent.parent / "dependencies" / "gncpy" / "src"))

from Vehicles.Vehicle import Vehicle
from gncpy.dynamics.basic import DoubleIntegrator
from gncpy.dynamics.aircraft import SimpleMultirotor
from ConfigurationSpace.ConfigSpace3D import ConfigurationSpace3D
from Utils.GeometryUtils import Box3D, PointXYZ


def test_double_integrator_vehicle():
    """Test Vehicle with DoubleIntegrator dynamics."""
    print("=" * 60)
    print("Testing Vehicle with DoubleIntegrator dynamics")
    print("=" * 60)
    
    # Create a double integrator vehicle (2D: x, y, vx, vy)
    # Pass the class and it will be instantiated
    vehicle = Vehicle(
        dynamics_class=DoubleIntegrator,
        size=(0.5, 0.5, 0.5),  # 0.5m cube
        initial_state=np.array([0.0, 0.0, 0.0, 0.0]),  # [x, y, vx, vy]
        state_indices={'position': [0, 1]}  # Position at indices 0,1 (x, y only)
    )
    
    print(f"Created: {vehicle}")
    print(f"Initial position (x,y): {vehicle.get_position()[:2]}")
    print(f"State: {vehicle.state.flatten()}")
    
    # Access the dynamics model directly
    print(f"\nDynamics model type: {type(vehicle.model).__name__}")
    print(f"State names: {vehicle.model.state_names}")
    
    # Propagate the vehicle
    dt = 0.1
    # For DoubleIntegrator with no control model, it just integrates velocity
    # We need to set initial velocity to see movement
    vehicle.state = np.array([[1.0], [0.5], [0.1], [0.05]])  # Give it some velocity
    
    print(f"\nPropagating with dt={dt}, initial velocity=[0.1, 0.05]")
    for i in range(5):
        vehicle.propagate(dt, u=None, state_args=(dt,))
        pos = vehicle.get_position()
        vel = vehicle.state[2:4].flatten()
        print(f"Step {i+1} - Position: ({pos[0]:.3f}, {pos[1]:.3f}), Velocity: ({vel[0]:.3f}, {vel[1]:.3f})")
    
    # Test collision checking with configuration space
    config_space = ConfigurationSpace3D([0, 10, 0, 10, 0, 10])
    obstacle = Box3D(center=PointXYZ(2, 2, 2), size=PointXYZ(1, 1, 1))
    config_space.add_obstacle(obstacle)
    
    collision = vehicle.check_collision_with_config_space(config_space)
    print(f"\nCollision detected: {collision}")
    
    distance = vehicle.get_nearest_obstacle_distance(config_space)
    print(f"Distance to nearest obstacle: {distance:.3f} m")
    
    print("\n✓ DoubleIntegrator test completed!\n")


def test_simple_multirotor_vehicle():
    """Test Vehicle with SimpleMultirotor dynamics (requires YAML config)."""
    print("=" * 60)
    print("Testing Vehicle with SimpleMultirotor dynamics")
    print("=" * 60)
    
    # Example of how you would create a multirotor vehicle
    # Uncomment when you have a valid config file
    
    yaml_path = Path(__file__).parent / "FastQuad.yaml"
    
    if yaml_path.exists() and yaml_path.stat().st_size > 0:
        try:
            vehicle = Vehicle(
                dynamics_class=SimpleMultirotor,
                size=(0.8, 0.8, 0.3),  # Quad dimensions
                params_file=str(yaml_path),  # Passed to SimpleMultirotor constructor
                state_indices={
                    'position': [4, 5, 6],  # NED position in SimpleMultirotor state
                    'rotation': [13, 14, 15]  # roll, pitch, yaw
                }
            )
            
            print(f"Created: {vehicle}")
            print(f"Model state map: {vehicle.model.state_map}")
            
            # Access dynamics directly
            print(f"Vehicle mass: {vehicle.model.vehicle.mass} kg")
            
            print("\n✓ SimpleMultirotor test completed!\n")
        except Exception as e:
            print(f"⚠ Could not create SimpleMultirotor (needs valid YAML): {e}\n")
    else:
        print("⚠ FastQuad.yaml not found or empty. Skipping SimpleMultirotor test.\n")


def test_already_instantiated_model():
    """Test Vehicle with an already-instantiated dynamics model."""
    print("=" * 60)
    print("Testing Vehicle with pre-instantiated dynamics model")
    print("=" * 60)
    
    # Create the dynamics model first
    dynamics_model = DoubleIntegrator()
    
    # Pass the instance directly (2D model: x, y, vx, vy)
    vehicle = Vehicle(
        dynamics_class=dynamics_model,  # Already instantiated
        size=(1.0, 0.5, 0.5),
        initial_state=np.array([[5.0], [5.0], [0.0], [0.0]]),  # [x, y, vx, vy]
        state_indices={'position': [0, 1]}  # Only x, y for 2D
    )
    
    print(f"Created: {vehicle}")
    print(f"Initial position (x,y): {vehicle.get_position()[:2]}")
    
    # Both vehicle.model and dynamics_model point to the same object
    print(f"Same model instance: {vehicle.model is dynamics_model}")
    
    print("\n✓ Pre-instantiated model test completed!\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("VEHICLE CLASS TESTS")
    print("=" * 60 + "\n")
    
    test_double_integrator_vehicle()
    test_simple_multirotor_vehicle()
    test_already_instantiated_model()
    
    print("=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)
    print("\nUsage pattern:")
    print("  vehicle = Vehicle(DynamicsClass, size=(...), params_file='...')")
    print("  vehicle.model.propagate_state(...)  # Access dynamics directly")
    print("  vehicle.box  # Use for collision checking")
    print("  vehicle.propagate(dt, u)  # Convenience method")
