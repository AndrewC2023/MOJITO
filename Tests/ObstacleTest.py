"""
Test script for the Obstacle system (StaticObstacle and SimpleMovingObstacle)
with ConfigurationSpace3D integration.
"""

import sys
from pathlib import Path
import numpy as np
import fcl

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from ConfigurationSpace.Obstacles import Obstacle, StaticObstacle, SimpleMovingObstacle
from ConfigurationSpace.ConfigSpace3D import ConfigurationSpace3D
from Utils.GeometryUtils import Box3D, PointXYZ


def test_static_obstacle():
    """Test static obstacle functionality."""
    print("\n=== Testing Static Obstacle ===")
    
    # Create FCL geometry for a box
    geometry = fcl.Box(1.0, 1.0, 1.0)
    
    # Create transform at position (3, 3, 3)
    transform = fcl.Transform(np.eye(3), [3.0, 3.0, 3.0])
    
    # Create static obstacle
    static_obs = StaticObstacle(geometry, transform)
    
    # Test that transform doesn't change with time
    t1 = static_obs.get_transform(0.0)
    t2 = static_obs.get_transform(10.0)
    t3 = static_obs.get_transform(100.0)
    
    print(f"Transform at t=0: {t1.getTranslation()}")
    print(f"Transform at t=10: {t2.getTranslation()}")
    print(f"Transform at t=100: {t3.getTranslation()}")
    
    assert np.allclose(t1.getTranslation(), [3.0, 3.0, 3.0])
    assert np.allclose(t2.getTranslation(), [3.0, 3.0, 3.0])
    assert np.allclose(t3.getTranslation(), [3.0, 3.0, 3.0])
    
    print("Pass! Static obstacle maintains constant position across time")


def test_simple_moving_obstacle():
    """Test simple moving obstacle with interpolation."""
    print("\n=== Testing Simple Moving Obstacle ===")
    
    # Create FCL geometry for a sphere
    geometry = fcl.Sphere(0.5)
    
    # Create trajectory: move from (0,0,0) at t=0 to (10,10,10) at t=10
    trajectory = [
        (fcl.Transform(np.eye(3), [0.0, 0.0, 0.0]), 0.0),
        (fcl.Transform(np.eye(3), [10.0, 10.0, 10.0]), 10.0)
    ]
    
    moving_obs = SimpleMovingObstacle(geometry, trajectory)
    
    # Test interpolation at various times
    t_start = moving_obs.get_transform(0.0)
    t_mid = moving_obs.get_transform(5.0)
    t_end = moving_obs.get_transform(10.0)
    
    print(f"Position at t=0: {t_start.getTranslation()}")
    print(f"Position at t=5 (interpolated): {t_mid.getTranslation()}")
    print(f"Position at t=10: {t_end.getTranslation()}")
    
    assert np.allclose(t_start.getTranslation(), [0.0, 0.0, 0.0])
    assert np.allclose(t_mid.getTranslation(), [5.0, 5.0, 5.0])
    assert np.allclose(t_end.getTranslation(), [10.0, 10.0, 10.0])
    
    print("Pass! Moving obstacle interpolates correctly")
    
    # Test time clamping (should print warnings)
    print("\nTesting time clamping (warnings expected):")
    t_before = moving_obs.get_transform(-5.0)
    t_after = moving_obs.get_transform(20.0)
    
    print(f"Position at t=-5 (clamped): {t_before.getTranslation()}")
    print(f"Position at t=20 (clamped): {t_after.getTranslation()}")
    
    assert np.allclose(t_before.getTranslation(), [0.0, 0.0, 0.0])
    assert np.allclose(t_after.getTranslation(), [10.0, 10.0, 10.0])
    
    print("Pass! Time clamping works correctly")


def test_configspace_integration():
    """Test integration of obstacles with ConfigurationSpace3D."""
    print("\n=== Testing ConfigSpace3D Integration ===")
    
    # Create configuration space
    config_space = ConfigurationSpace3D([0, 10, 0, 10, 0, 10])
    
    # Add a static obstacle at (3, 3, 3)
    static_geom = fcl.Box(1.0, 1.0, 1.0)
    static_transform = fcl.Transform(np.eye(3), [3.0, 3.0, 3.0])
    static_obs = StaticObstacle(static_geom, static_transform)
    config_space.add_obstacle(static_obs)
    
    # Add a moving obstacle that moves from (7, 7, 7) at t=0 to (8, 8, 8) at t=10
    moving_geom = fcl.Sphere(0.5)
    moving_trajectory = [
        (fcl.Transform(np.eye(3), [7.0, 7.0, 7.0]), 0.0),
        (fcl.Transform(np.eye(3), [8.0, 8.0, 8.0]), 10.0)
    ]
    moving_obs = SimpleMovingObstacle(moving_geom, moving_trajectory)
    config_space.add_obstacle(moving_obs)
    
    print(f"Added {config_space.get_num_obstacles()} obstacles to config space")
    
    # Create a test vehicle using FCL directly
    vehicle_geom = fcl.Box(0.5, 0.5, 0.5)
    vehicle_transform = fcl.Transform(np.eye(3), [5.0, 5.0, 5.0])
    vehicle = fcl.CollisionObject(vehicle_geom, vehicle_transform)
    
    # Test collision at t=0 (should be free)
    collision_t0 = config_space.check_collision(vehicle, t=0.0)
    print(f"Collision at t=0 (vehicle at 5,5,5): {collision_t0}")
    assert not collision_t0, "Vehicle should be collision-free at (5,5,5)"
    
    # Move vehicle to collide with static obstacle
    vehicle_transform = fcl.Transform(np.eye(3), [3.0, 3.0, 3.0])
    vehicle.setTransform(vehicle_transform)
    collision_static = config_space.check_collision(vehicle, t=0.0)
    print(f"Collision with static obstacle (vehicle at 3,3,3): {collision_static}")
    assert collision_static, "Vehicle should collide with static obstacle"
    
    # Move vehicle to collide with moving obstacle at t=0
    vehicle_transform = fcl.Transform(np.eye(3), [7.0, 7.0, 7.0])
    vehicle.setTransform(vehicle_transform)
    collision_moving_t0 = config_space.check_collision(vehicle, t=0.0)
    print(f"Collision with moving obstacle at t=0 (vehicle at 7,7,7): {collision_moving_t0}")
    assert collision_moving_t0, "Vehicle should collide with moving obstacle at t=0"
    
    # Check same position at t=10 (moving obstacle moved slightly)
    collision_moving_t10 = config_space.check_collision(vehicle, t=10.0)
    print(f"Collision with moving obstacle at t=10 (vehicle at 7,7,7): {collision_moving_t10}")
    
    print("Pass! ConfigSpace3D integration works correctly with time-dependent collisions")
    
    # Test detailed collision query
    vehicle_transform = fcl.Transform(np.eye(3), [5.0, 5.0, 5.0])
    vehicle.setTransform(vehicle_transform)
    result = config_space.query_collision_detailed(vehicle, t=5.0)
    print(f"\nDetailed query at t=5.0:")
    print(f"  Has collision: {result.has_collision}")
    print(f"  Number of obstacle collisions: {result.num_collisions}")
    print(f"  Min obstacle distance: {result.min_obstacle_distance:.3f}")
    print(f"  Out of bounds: {result.is_out_of_bounds}")
    
    for obs_data in result.obstacle_data:
        print(f"  Obstacle {obs_data.obstacle_index}: "
              f"distance={obs_data.distance:.3f}, "
              f"collision={obs_data.is_collision}")
    
    print("Pass! Detailed collision query works with time parameter")


def test_complex_trajectory():
    """Test moving obstacle with multiple waypoints."""
    print("\n=== Testing Complex Trajectory ===")
    
    geometry = fcl.Box(0.5, 0.5, 0.5)
    
    # Create trajectory with multiple waypoints
    trajectory = [
        (fcl.Transform(np.eye(3), [0.0, 0.0, 0.0]), 0.0),
        (fcl.Transform(np.eye(3), [5.0, 0.0, 0.0]), 2.0),
        (fcl.Transform(np.eye(3), [5.0, 5.0, 0.0]), 4.0),
        (fcl.Transform(np.eye(3), [0.0, 5.0, 0.0]), 6.0),
        (fcl.Transform(np.eye(3), [0.0, 0.0, 0.0]), 8.0),
    ]
    
    moving_obs = SimpleMovingObstacle(geometry, trajectory)
    
    # Test positions at various times
    test_times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    
    print("Trajectory positions:")
    for t in test_times:
        pos = moving_obs.get_transform(t).getTranslation()
        print(f"  t={t:.1f}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
    
    # Verify interpolation at t=1.0 (halfway between waypoints 0 and 1)
    pos_t1 = moving_obs.get_transform(1.0).getTranslation()
    assert np.allclose(pos_t1, [2.5, 0.0, 0.0], atol=0.01), "Interpolation error at t=1.0"
    
    print("Pass! Complex trajectory with multiple waypoints works correctly")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Obstacle System")
    print("=" * 60)
    
    test_static_obstacle()
    test_simple_moving_obstacle()
    test_configspace_integration()
    test_complex_trajectory()
    
    print("\n" + "=" * 60)
    print("All tests passed! Pass!")
    print("=" * 60)
