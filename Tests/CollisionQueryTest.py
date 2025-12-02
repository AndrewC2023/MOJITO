"""
Test the detailed collision query functionality for cost function optimization.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from ConfigurationSpace.ConfigSpace3D import ConfigurationSpace3D
from Utils.GeometryUtils import Box3D, PointXYZ

def test_collision_query():
    print("Testing detailed collision query...\n")
    
    # Create configuration space
    config_space = ConfigurationSpace3D([0, 10, 0, 10, 0, 10])
    
    # Add obstacles
    obstacle1 = Box3D(center=PointXYZ(3, 3, 5), size=PointXYZ(1, 1, 2))
    obstacle2 = Box3D(center=PointXYZ(7, 7, 5), size=PointXYZ(1.5, 1.5, 2))
    obstacle3 = Box3D(center=PointXYZ(5, 5, 8), size=PointXYZ(0.5, 0.5, 0.5))
    
    config_space.add_obstacles([obstacle1, obstacle2, obstacle3])
    
    # Test scenarios
    test_cases = [
        ("Far from all obstacles", PointXYZ(1, 1, 2), PointXYZ(0.3, 0.3, 0.3)),
        ("Close to obstacle 1", PointXYZ(3.8, 3, 5), PointXYZ(0.3, 0.3, 0.3)),
        ("Colliding with obstacle 1", PointXYZ(3, 3, 5), PointXYZ(0.6, 0.6, 1)),
        ("Colliding with multiple", PointXYZ(4, 4, 5.5), PointXYZ(2, 2, 2)),
        ("Out of bounds", PointXYZ(-1, 5, 5), PointXYZ(0.3, 0.3, 0.3)),
    ]
    
    for description, center, size in test_cases:
        print(f"=== {description} ===")
        vehicle = Box3D(center=center, size=size)
        
        result = config_space.query_collision_detailed(vehicle)
        
        print(f"Position: ({center.x}, {center.y}, {center.z})")
        print(f"Has collision: {result.has_collision}")
        print(f"Out of bounds: {result.is_out_of_bounds}")
        print(f"Number of collisions: {result.num_collisions}")
        print(f"Total penetration depth: {result.total_penetration_depth:.4f}")
        print(f"Min obstacle distance: {result.min_obstacle_distance:.4f}")
        
        print("\nPer-obstacle data:")
        for obs in result.obstacle_data:
            status = "COLLISION" if obs.is_collision else "clear"
            print(f"  Obstacle {obs.obstacle_index}: {status}, "
                  f"distance={obs.distance:.4f}, penetration={obs.penetration_depth:.4f}")
        
        # Example cost function calculation
        # Penalize collisions heavily, but also penalize proximity to encourage safe distances
        collision_cost = result.total_penetration_depth * 100.0  # Heavy penalty for penetration
        
        # Proximity cost: penalize being within 1.0 unit of obstacles
        proximity_cost = 0.0
        for obs in result.obstacle_data:
            if not obs.is_collision and obs.distance < 1.0:
                proximity_cost += (1.0 - obs.distance) * 10.0  # Smooth penalty
        
        boundary_cost = 100.0 if result.is_out_of_bounds else 0.0
        
        total_cost = collision_cost + proximity_cost + boundary_cost
        
        print(f"\nCost breakdown:")
        print(f"  Collision cost: {collision_cost:.2f}")
        print(f"  Proximity cost: {proximity_cost:.2f}")
        print(f"  Boundary cost: {boundary_cost:.2f}")
        print(f"  TOTAL COST: {total_cost:.2f}")
        print()

if __name__ == "__main__":
    test_collision_query()
