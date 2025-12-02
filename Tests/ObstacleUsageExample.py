"""
Example usage of the Obstacle system with ConfigurationSpace3D.

This demonstrates how to:
1. Create static obstacles
2. Create simple moving obstacles with trajectories
3. Add obstacles to a configuration space
4. Check collisions at different times
"""

import sys
from pathlib import Path
import numpy as np
import fcl

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from ConfigurationSpace.Obstacles import StaticObstacle, SimpleMovingObstacle
from ConfigurationSpace.ConfigSpace3D import ConfigurationSpace3D
from Utils.GeometryUtils import Box3D, PointXYZ


def main():
    print("Obstacle System Usage Example")
    print("=" * 60)
    
    # Create a 3D configuration space (10x10x10 meters)
    config_space = ConfigurationSpace3D([0, 10, 0, 10, 0, 10])
    print(f"Created config space: x[0-10], y[0-10], z[0-10]\n")
    
    # Example 1: Add a static obstacle (e.g., a building)
    print("1. Adding static obstacle (building at 3,3,3)...")
    building_geometry = fcl.Box(2.0, 2.0, 3.0)  # 2x2x3 meter building
    building_transform = fcl.Transform(np.eye(3), [3.0, 3.0, 1.5])
    building = StaticObstacle(building_geometry, building_transform)
    config_space.add_obstacle(building)
    print(f"   Static obstacle added\n")
    
    # Example 2: Add a simple moving obstacle (e.g., another drone)
    print("2. Adding moving obstacle (drone patrol route)...")
    
    # Define a square patrol pattern
    drone_geometry = fcl.Sphere(0.3)  # 30cm radius drone
    patrol_trajectory = [
        (fcl.Transform(np.eye(3), [7.0, 7.0, 5.0]), 0.0),   # Start corner
        (fcl.Transform(np.eye(3), [9.0, 7.0, 5.0]), 2.0),   # Move right
        (fcl.Transform(np.eye(3), [9.0, 9.0, 5.0]), 4.0),   # Move forward
        (fcl.Transform(np.eye(3), [7.0, 9.0, 5.0]), 6.0),   # Move left
        (fcl.Transform(np.eye(3), [7.0, 7.0, 5.0]), 8.0),   # Return to start
    ]
    patrol_drone = SimpleMovingObstacle(drone_geometry, patrol_trajectory)
    config_space.add_obstacle(patrol_drone)
    print(f"   Moving obstacle with {len(patrol_trajectory)} waypoints added\n")
    
    # Example 3: Check collision for our vehicle at different times
    print("3. Checking collisions for vehicle trajectory...")
    # Create vehicle using FCL directly
    vehicle_geometry = fcl.Box(0.5, 0.5, 0.5)
    vehicle_transform = fcl.Transform(np.eye(3), [5.0, 5.0, 5.0])
    vehicle = fcl.CollisionObject(vehicle_geometry, vehicle_transform)
    
    # Safe position at t=0
    collision = config_space.check_collision(vehicle, t=0.0)
    print(f"   t=0.0s, vehicle at (5,5,5): collision={collision}")
    
    # Move vehicle near the patrol drone at different times
    vehicle_transform = fcl.Transform(np.eye(3), [8.0, 7.0, 5.0])
    vehicle.setTransform(vehicle_transform)
    
    collision_t0 = config_space.check_collision(vehicle, t=0.0)
    collision_t1 = config_space.check_collision(vehicle, t=1.0)
    collision_t2 = config_space.check_collision(vehicle, t=2.0)
    
    print(f"   t=0.0s, vehicle at (8,7,5): collision={collision_t0}")
    print(f"   t=1.0s, vehicle at (8,7,5): collision={collision_t1}")
    print(f"   t=2.0s, vehicle at (8,7,5): collision={collision_t2}")
    print()
    
    # Example 4: Get detailed collision information for cost functions
    print("4. Getting detailed collision information...")
    vehicle_transform = fcl.Transform(np.eye(3), [5.0, 5.0, 5.0])
    vehicle.setTransform(vehicle_transform)
    
    result = config_space.query_collision_detailed(vehicle, t=3.0)
    print(f"   Has collision: {result.has_collision}")
    print(f"   Out of bounds: {result.is_out_of_bounds}")
    print(f"   Number of collisions: {result.num_collisions}")
    print(f"   Total penetration depth: {result.total_penetration_depth:.3f}m")
    print(f"   Min obstacle distance: {result.min_obstacle_distance:.3f}m")
    print()
    
    for i, obs_data in enumerate(result.obstacle_data):
        obs_type = "Static" if i == 0 else "Moving"
        print(f"   {obs_type} obstacle {i}:")
        print(f"     - Distance: {obs_data.distance:.3f}m")
        print(f"     - Collision: {obs_data.is_collision}")
        if obs_data.is_collision:
            print(f"     - Penetration: {obs_data.penetration_depth:.3f}m")
    print()
    
    # Example 5: Check distance to nearest obstacle over time
    print("5. Tracking nearest obstacle distance over time...")
    print("   Vehicle positioned at (8, 7, 5) - near patrol drone path")
    vehicle_transform = fcl.Transform(np.eye(3), [8.0, 7.0, 5.0])
    vehicle.setTransform(vehicle_transform)
    
    for t in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]:
        distance = config_space.get_nearest_obstacle_distance(vehicle, t=t)
        
        # Get the patrol drone position at this time for reference
        patrol_pos = patrol_drone.get_transform(t).getTranslation()
        
        print(f"   t={t:.1f}s: nearest = {distance:.3f}m "
              f"(patrol drone at [{patrol_pos[0]:.1f}, {patrol_pos[1]:.1f}, {patrol_pos[2]:.1f}])")
    
    print("\n6. Moving vehicle along a path to show dynamic distance changes...")
    print("   Vehicle moves from (5,5,5) toward patrol drone")
    
    # Move vehicle along a path that gets closer to the moving drone
    vehicle_positions = [
        ([5.0, 5.0, 5.0], "center of space"),
        ([6.0, 6.0, 5.0], "moving toward patrol"),
        ([7.0, 7.0, 5.0], "at patrol start point"),
        ([8.0, 7.0, 5.0], "following patrol drone"),
    ]
    
    for pos, description in vehicle_positions:
        vehicle_transform = fcl.Transform(np.eye(3), pos)
        vehicle.setTransform(vehicle_transform)
        distance = config_space.get_nearest_obstacle_distance(vehicle, t=0.0)
        print(f"   Vehicle at ({pos[0]}, {pos[1]}, {pos[2]}) [{description}]: {distance:.3f}m")
    
    print("\n" + "=" * 60)
    print("Example complete!")


if __name__ == "__main__":
    main()
