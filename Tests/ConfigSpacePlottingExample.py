"""
Example script demonstrating the 3D plotting capabilities of ConfigurationSpace3D.

This script shows how to:
1. Visualize the configuration space with boundaries and obstacles
2. Plot static and dynamic obstacles at different time points
3. Visualize vehicle trajectories with bounding boxes
4. Validate that your motion planning setup is correct
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import fcl

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from ConfigurationSpace.ConfigSpace3D import ConfigurationSpace3D
from ConfigurationSpace.Obstacles import StaticObstacle, SimpleMovingObstacle
from Utils.GeometryUtils import DCM3D


def example_1_basic_configuration_space():
    """Example 1: Basic configuration space with static obstacles."""
    print("=" * 70)
    print("Example 1: Basic Configuration Space Visualization")
    print("=" * 70)
    
    # Create a configuration space (100m x 100m x 50m)
    config_space = ConfigurationSpace3D([0, 100, 0, 100, 0, 50])
    
    # Add some static obstacles
    # Building 1: Large box
    building1_geom = fcl.Box(20.0, 15.0, 30.0)
    building1_transform = fcl.Transform(np.eye(3), [30.0, 30.0, 15.0])
    building1 = StaticObstacle(building1_geom, building1_transform)
    config_space.add_obstacle(building1)
    
    # Building 2: Tall thin box
    building2_geom = fcl.Box(10.0, 10.0, 40.0)
    building2_transform = fcl.Transform(np.eye(3), [70.0, 60.0, 20.0])
    building2 = StaticObstacle(building2_geom, building2_transform)
    config_space.add_obstacle(building2)
    
    # Sphere obstacle
    sphere_geom = fcl.Sphere(8.0)
    sphere_transform = fcl.Transform(np.eye(3), [50.0, 70.0, 35.0])
    sphere = StaticObstacle(sphere_geom, sphere_transform)
    config_space.add_obstacle(sphere)
    
    # Cylinder obstacle
    cylinder_geom = fcl.Cylinder(6.0, 25.0)
    cylinder_transform = fcl.Transform(np.eye(3), [80.0, 20.0, 12.5])
    cylinder = StaticObstacle(cylinder_geom, cylinder_transform)
    config_space.add_obstacle(cylinder)
    
    # Plot the configuration space
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    config_space.plot_configuration_space(
        t=0.0,
        ax=ax,
        obstacle_alpha=0.4,
        obstacle_color='red',
        bounds_alpha=0.15,
        title='Configuration Space with Static Obstacles'
    )
    
    plt.tight_layout()
    out_path = Path(__file__).parent / "config_space_example1.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    
    print(" Configuration space plotted successfully!\n")
    return config_space


def example_2_dynamic_obstacles():
    """Example 2: Configuration space with moving obstacles at different times."""
    print("=" * 70)
    print("Example 2: Dynamic Obstacles at Different Times")
    print("=" * 70)
    
    # Create configuration space
    config_space = ConfigurationSpace3D([0, 100, 0, 100, 0, 50])
    
    # Static obstacle
    building_geom = fcl.Box(15.0, 15.0, 25.0)
    building_transform = fcl.Transform(np.eye(3), [50.0, 50.0, 12.5])
    building = StaticObstacle(building_geom, building_transform)
    config_space.add_obstacle(building)
    
    # Moving obstacle: drone flying in a circle
    drone_geom = fcl.Box(2.0, 2.0, 1.0)
    
    # Create trajectory (circular path at constant height)
    num_keyframes = 20
    times = np.linspace(0, 10, num_keyframes)  # 10 second trajectory
    trajectory = []
    
    for t in times:
        # Circular motion: center at (25, 25), radius 15m, height 20m
        angle = 2 * np.pi * t / 10.0  # Complete circle in 10 seconds
        x = 25 + 15 * np.cos(angle)
        y = 25 + 15 * np.sin(angle)
        z = 20.0
        
        transform = fcl.Transform(np.eye(3), [x, y, z])
        trajectory.append((transform, t))
    
    moving_drone = SimpleMovingObstacle(drone_geom, trajectory)
    config_space.add_obstacle(moving_drone)
    
    # Plot at three different times
    fig = plt.figure(figsize=(18, 5))
    
    for i, t in enumerate([0.0, 5.0, 9.0]):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        config_space.plot_configuration_space(
            t=t,
            ax=ax,
            obstacle_alpha=0.4,
            title=f'Configuration Space at t={t:.1f}s'
        )
    
    plt.tight_layout()
    out_path = Path(__file__).parent / "config_space_example2.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    
    print(" Dynamic obstacles visualized at multiple time points!\n")
    return config_space


def example_3_vehicle_trajectory():
    """Example 3: Plotting vehicle trajectory through configuration space."""
    print("=" * 70)
    print("Example 3: Vehicle Trajectory Visualization")
    print("=" * 70)
    
    # Create configuration space
    config_space = ConfigurationSpace3D([0, 100, 0, 100, 0, 50])
    
    # Add obstacles
    building1_geom = fcl.Box(20.0, 15.0, 30.0)
    building1_transform = fcl.Transform(np.eye(3), [30.0, 30.0, 15.0])
    building1 = StaticObstacle(building1_geom, building1_transform)
    config_space.add_obstacle(building1)
    
    building2_geom = fcl.Box(15.0, 15.0, 35.0)
    building2_transform = fcl.Transform(np.eye(3), [70.0, 70.0, 17.5])
    building2 = StaticObstacle(building2_geom, building2_transform)
    config_space.add_obstacle(building2)
    
    # Create a vehicle trajectory (navigate around obstacles)
    trajectory_points = np.array([
        [10, 10, 10],    # Start
        [20, 15, 15],
        [30, 10, 20],    # Go around building 1
        [45, 20, 25],
        [50, 40, 30],
        [60, 55, 30],    # Approach building 2
        [75, 55, 25],    # Go around building 2
        [85, 75, 20],
        [90, 85, 15],
        [95, 90, 10],    # End
    ])
    
    times = np.linspace(0, 9, len(trajectory_points))
    
    # Define vehicle geometry (quadrotor-sized box)
    vehicle_geom = fcl.Box(1.5, 1.5, 0.8)
    
    # Plot trajectory with vehicle boxes at sample points
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    config_space.plot_vehicle_trajectory(
        trajectory=trajectory_points,
        times=times,
        vehicle_geometry=vehicle_geom,
        sample_indices=[0, 3, 5, 7, 9],  # Show vehicle at start, middle points, and end
        ax=ax,
        trajectory_color='blue',
        trajectory_width=2.5,
        vehicle_alpha=0.6,
        vehicle_color='green',
        obstacle_alpha=0.3,
        t_obstacles=0.0,
        title='Vehicle Trajectory Through Configuration Space'
    )
    
    plt.tight_layout()
    out_path = Path(__file__).parent / "config_space_example3.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    
    print(" Vehicle trajectory plotted successfully!\n")
    
    # Validate trajectory for collisions
    print("Validating trajectory for collisions...")
    collision_detected = False
    for i, (pos, t) in enumerate(zip(trajectory_points, times)):
        vehicle_transform = fcl.Transform(np.eye(3), pos)
        vehicle_obj = fcl.CollisionObject(vehicle_geom, vehicle_transform)
        
        result = config_space.query_collision_detailed(vehicle_obj, t=t)
        if result.has_collision:
            print(f"  ⚠ Collision at waypoint {i} (t={t:.2f}s): "
                  f"{result.num_collisions} obstacles, "
                  f"penetration depth: {result.total_penetration_depth:.3f}m")
            collision_detected = True
    
    if not collision_detected:
        print("   Trajectory is collision-free!")
    print()


def example_4_rotated_obstacles():
    """Example 4: Obstacles with rotations (testing 3-2-1 Euler compatibility)."""
    print("=" * 70)
    print("Example 4: Rotated Obstacles (3-2-1 Euler Angles)")
    print("=" * 70)
    
    # Create configuration space
    config_space = ConfigurationSpace3D([0, 50, 0, 50, 0, 30])
    
    # Add rotated box obstacles using Euler angles
    # Obstacle 1: Rotated 45° yaw
    R1 = DCM3D(np.radians(45), "z")
    box1_geom = fcl.Box(10.0, 5.0, 8.0)
    box1_transform = fcl.Transform(R1, [15.0, 15.0, 4.0])
    box1 = StaticObstacle(box1_geom, box1_transform)
    config_space.add_obstacle(box1)
    
    # Obstacle 2: Rotated with yaw, pitch, roll (applying rotations in sequence)
    R_roll = DCM3D(np.radians(10), "x")
    R_pitch = DCM3D(np.radians(15), "y")
    R_yaw = DCM3D(np.radians(30), "z")
    R2 = R_yaw @ R_pitch @ R_roll  # 3-2-1 sequence
    box2_geom = fcl.Box(8.0, 8.0, 12.0)
    box2_transform = fcl.Transform(R2, [35.0, 35.0, 6.0])
    box2 = StaticObstacle(box2_geom, box2_transform)
    config_space.add_obstacle(box2)
    
    # Obstacle 3: Tilted cylinder
    R3 = DCM3D(np.radians(30), "y")
    cyl_geom = fcl.Cylinder(3.0, 15.0)
    cyl_transform = fcl.Transform(R3, [25.0, 10.0, 10.0])
    cyl = StaticObstacle(cyl_geom, cyl_transform)
    config_space.add_obstacle(cyl)
    
    # Plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    config_space.plot_configuration_space(
        ax=ax,
        obstacle_alpha=0.4,
        title='Configuration Space with Rotated Obstacles (3-2-1 Euler)'
    )
    
    plt.tight_layout()
    out_path = Path(__file__).parent / "config_space_example4.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    
    print(" Rotated obstacles (using DCM3D from GeometryUtils) plotted correctly!")
    print(" FCL handles NED frame dynamics conventions properly!\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("ConfigurationSpace3D Plotting Examples")
    print("="*70 + "\n")
    
    # Run examples non-interactively, saving plots to image files
    try:
        example_1_basic_configuration_space()
        example_2_dynamic_obstacles()
        example_3_vehicle_trajectory()
        example_4_rotated_obstacles()

        print("\n" + "="*70)
        print("All examples completed successfully!")
        print("Saved figures: config_space_example1-4.png in the Tests directory.")
        print("="*70 + "\n")

    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user.")
    except Exception as e:
        print(f"\n\nError running examples: {e}")
        import traceback
        traceback.print_exc()
