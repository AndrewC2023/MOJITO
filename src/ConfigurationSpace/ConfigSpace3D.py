import numpy as np
from typing import List, Dict, NamedTuple
import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.append(str(Path(__file__).parent.parent))

from Utils.GeometryUtils import Box3D, PointXYZ


class ObstacleProximity(NamedTuple):
    """
    Information about proximity to a single obstacle.
    
    Attributes:
        obstacle_index: Index of the obstacle in the config space obstacle list
        distance: Distance to obstacle (negative if colliding, positive if separated)
        is_collision: True if box is colliding with this obstacle
        penetration_depth: Depth of penetration if colliding (0 if not colliding)
    """
    obstacle_index: int
    distance: float
    is_collision: bool
    penetration_depth: float


class CollisionQueryResult(NamedTuple):
    """
    Complete result of a collision query against configuration space.
    
    This structure provides all information needed for smooth, continuous cost functions
    in optimization (MPC, PSO, GA, etc). It allows you to:
    
    1. Heavily penalize actual collisions using total_penetration_depth
       - Accounts for obstacle size (grazing a pebble vs hitting a wall)
    
    2. Create smooth proximity costs to encourage safe distances
       - Access distance to each obstacle even when not colliding
       - Implement gradual penalties as vehicle approaches obstacles
    
    3. Handle boundary violations with is_out_of_bounds flag
    
    Example cost function:
        result = config_space.query_collision_detailed(vehicle.box)
        
        # Collision penalty (discontinuous at 0, but depth provides gradient)
        collision_cost = result.total_penetration_depth * 1000.0
        
        # Proximity penalty (smooth, continuous gradient)
        proximity_cost = 0.0
        for obs in result.obstacle_data:
            if not obs.is_collision and obs.distance < safe_distance:
                proximity_cost += (safe_distance - obs.distance) ** 2
        
        # Boundary penalty
        boundary_cost = 1000.0 if result.is_out_of_bounds else 0.0
        
        total_cost = collision_cost + proximity_cost + boundary_cost
    
    Attributes:
        has_collision: True if any collision detected (obstacles or boundaries)
        is_out_of_bounds: True if outside boundary limits
        obstacle_data: List of ObstacleProximity for each obstacle
        total_penetration_depth: Sum of all penetration depths (for cost function)
        num_collisions: Number of obstacles being collided with
        min_obstacle_distance: Distance to nearest obstacle (inf if none)
    """
    has_collision: bool
    is_out_of_bounds: bool
    obstacle_data: List[ObstacleProximity]
    total_penetration_depth: float
    num_collisions: int
    min_obstacle_distance: float

class ConfigurationSpace3D:
    """
    3D Configuration Space for motion planning with Box3D obstacles.
    
    Manages a bounded 3D space with obstacles represented as Box3D objects.
    Provides methods to add/remove obstacles and check for collisions.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize a 3D configuration space.
        
        Args:
            *args: Can pass dimensions as [xMin, xMax, yMin, yMax, zMin, zMax]
            **kwargs: Can pass 'dimensions' keyword argument
        """
        # Set defaults
        self.xMin = -10
        self.xMax = 10
        self.yMin = -10
        self.yMax = 10
        self.zMin = -10
        self.zMax = 10
        
        # Process positional arguments
        if len(args) >= 1:
            # First argument: dimensions [xMin, xMax, yMin, yMax, zMin, zMax]
            dimensions = args[0]
            if len(dimensions) == 6:
                self.xMin = dimensions[0]
                self.xMax = dimensions[1]
                self.yMin = dimensions[2]
                self.yMax = dimensions[3]
                self.zMin = dimensions[4]
                self.zMax = dimensions[5]
            else:
                raise ValueError("Dimensions must have 6 values: [xMin, xMax, yMin, yMax, zMin, zMax]")
            
        if len(args) > 1:
            raise ValueError("Too many positional arguments. Expected at most 1.")
        
        # Process keyword arguments (override positional if specified)
        if "dimensions" in kwargs:
            dimensions = kwargs["dimensions"]
            if len(dimensions) >= 6:
                self.xMin = dimensions[0]
                self.xMax = dimensions[1]
                self.yMin = dimensions[2]
                self.yMax = dimensions[3]
                self.zMin = dimensions[4]
                self.zMax = dimensions[5]
            else:
                raise ValueError("Dimensions must have at least 6 values: [xMin, xMax, yMin, yMax, zMin, zMax]")
        
        # Initialize obstacle list
        self.obstacles: List[Box3D] = []
        
        # Create boundary box for the configuration space
        self._update_boundary_box()
    
    def _update_boundary_box(self):
        """Update the boundary box representing the field bounds."""
        center = PointXYZ(
            (self.xMin + self.xMax) / 2,
            (self.yMin + self.yMax) / 2,
            (self.zMin + self.zMax) / 2
        )
        size = PointXYZ(
            self.xMax - self.xMin,
            self.yMax - self.yMin,
            self.zMax - self.zMin
        )
        self.boundary_box = Box3D(center, size)
    
    def reconfigure(self, dimensions: List[float]):
        """
        Reconfigure the dimensions of the configuration space.
        
        Args:
            dimensions: List of 6 floats [xMin, xMax, yMin, yMax, zMin, zMax]
        """
        if len(dimensions) != 6:
            raise ValueError("Dimensions must have 6 values: [xMin, xMax, yMin, yMax, zMin, zMax]")
        
        self.xMin = dimensions[0]
        self.xMax = dimensions[1]
        self.yMin = dimensions[2]
        self.yMax = dimensions[3]
        self.zMin = dimensions[4]
        self.zMax = dimensions[5]
        self._update_boundary_box()
        self.obstacles.clear()
    
    def add_obstacle(self, obstacle: Box3D):
        """
        Add a Box3D obstacle to the configuration space.
        
        Args:
            obstacle: Box3D object to add as an obstacle
        """
        if not isinstance(obstacle, Box3D):
            raise ValueError("Obstacle must be a Box3D object")
        self.obstacles.append(obstacle)
    
    def add_obstacles(self, obstacles: List[Box3D]):
        """
        Add multiple Box3D obstacles to the configuration space.
        
        Args:
            obstacles: List of Box3D objects to add as obstacles
        """
        for obstacle in obstacles:
            self.add_obstacle(obstacle)
    
    def remove_obstacle(self, obstacle: Box3D):
        """
        Remove a specific obstacle from the configuration space.
        
        Args:
            obstacle: Box3D object to remove
        """
        if obstacle in self.obstacles:
            self.obstacles.remove(obstacle)
    
    def remove_obstacle_at_index(self, index: int):
        """
        Remove an obstacle at a specific index.
        
        Args:
            index: Index of the obstacle to remove
        """
        if 0 <= index < len(self.obstacles):
            self.obstacles.pop(index)
        else:
            raise IndexError(f"Obstacle index {index} out of range (0 to {len(self.obstacles)-1})")
    
    def clear_obstacles(self):
        """Remove all obstacles from the configuration space."""
        self.obstacles.clear()
    
    def is_point_in_bounds(self, point: PointXYZ) -> bool:
        """
        Check if a point is within the configuration space bounds.
        
        Args:
            point: PointXYZ to check
            
        Returns:
            bool: True if point is within bounds, False otherwise
        """
        return (self.xMin <= point.x <= self.xMax and
                self.yMin <= point.y <= self.yMax and
                self.zMin <= point.z <= self.zMax)
    
    def is_box_in_bounds(self, box: Box3D) -> bool:
        """
        Check if a box is entirely within the configuration space bounds.
        
        Args:
            box: Box3D object to check
            
        Returns:
            bool: True if box is entirely within bounds, False otherwise
        """
        # Fast AABB check: use box center and half-extents
        # This is much faster than checking all 8 vertices for axis-aligned or near-axis-aligned boxes
        cx, cy, cz = box.center.x, box.center.y, box.center.z
        
        # For rotated boxes, we need the actual extent in each axis
        # Conservative approach: use half diagonal as radius for quick rejection
        half_diagonal = 0.5 * np.sqrt(box.size.x**2 + box.size.y**2 + box.size.z**2)
        
        # Check if center ± half_diagonal is within bounds
        if (cx - half_diagonal < self.xMin or cx + half_diagonal > self.xMax or
            cy - half_diagonal < self.yMin or cy + half_diagonal > self.yMax or
            cz - half_diagonal < self.zMin or cz + half_diagonal > self.zMax):
            return False
        
        return True
    
    def check_collision(self, box: Box3D) -> bool:
        """
        Check if a box collides with field boundaries or any obstacles.
        
        Args:
            box: Box3D object to check for collision
            
        Returns:
            bool: True if collision detected, False otherwise
        """
        # Check if box is out of bounds
        if not self.is_box_in_bounds(box):
            return True
        
        # Early exit if no obstacles
        if not self.obstacles:
            return False
        
        # Check collision with all obstacles (loop unrolling not needed, Python JIT handles it)
        for obstacle in self.obstacles:
            if box.check_collision(obstacle):
                return True
        
        return False
    
    def check_collision_with_obstacles_only(self, box: Box3D) -> bool:
        """
        Check if a box collides with any obstacles (ignores boundaries).
        
        Args:
            box: Box3D object to check for collision
            
        Returns:
            bool: True if collision with obstacle detected, False otherwise
        """
        for obstacle in self.obstacles:
            if box.check_collision(obstacle):
                return True
        return False
    
    def get_nearest_obstacle_distance(self, box: Box3D) -> float:
        """
        Get the distance to the nearest obstacle.
        
        Args:
            box: Box3D object to check distance from
            
        Returns:
            float: Minimum distance to nearest obstacle (inf if no obstacles)
        """
        if not self.obstacles:
            return float('inf')
        
        min_distance = float('inf')
        for obstacle in self.obstacles:
            distance = box.distance_to(obstacle)
            # Early exit if already in collision
            if distance <= 0:
                return 0.0
            if distance < min_distance:
                min_distance = distance
        
        return min_distance
    
    def get_distance_to_boundaries(self, box: Box3D) -> float:
        """
        Get the distance from a box to the nearest boundary wall.
        
        Args:
            box: Box3D object to check distance from
            
        Returns:
            float: Minimum distance to nearest boundary (negative if outside bounds)
        """
        cx, cy, cz = box.center.x, box.center.y, box.center.z
        
        # For rotated boxes, use half diagonal as conservative radius
        half_diagonal = 0.5 * np.sqrt(box.size.x**2 + box.size.y**2 + box.size.z**2)
        
        # Distance to each boundary wall
        dist_to_xmin = cx - half_diagonal - self.xMin
        dist_to_xmax = self.xMax - (cx + half_diagonal)
        dist_to_ymin = cy - half_diagonal - self.yMin
        dist_to_ymax = self.yMax - (cy + half_diagonal)
        dist_to_zmin = cz - half_diagonal - self.zMin
        dist_to_zmax = self.zMax - (cz + half_diagonal)
        
        # Return minimum distance (negative if out of bounds)
        return min(dist_to_xmin, dist_to_xmax, dist_to_ymin, dist_to_ymax, dist_to_zmin, dist_to_zmax)
    
    def query_collision_detailed(self, box: Box3D) -> CollisionQueryResult:
        """
        Perform comprehensive collision query with detailed information for cost functions.
        
        Returns information about all obstacles including:
        - Which obstacles are being collided with
        - Penetration depth for each collision
        - Distance to each obstacle (even non-colliding ones)
        - Total penetration depth (sum of all collisions)
        
        This provides smooth, continuous data for optimization cost functions.
        
        Args:
            box: Box3D object to query
            
        Returns:
            CollisionQueryResult with complete proximity information
        """
        # Check boundary collision
        is_out_of_bounds = not self.is_box_in_bounds(box)
        
        # Collect data for each obstacle
        obstacle_data = []
        total_penetration = 0.0
        num_collisions = 0
        min_distance = float('inf')
        
        for idx, obstacle in enumerate(self.obstacles):
            # Get distance (negative if penetrating)
            distance = box.distance_to(obstacle)
            
            # Determine collision and penetration depth
            is_collision = distance <= 0
            penetration = abs(distance) if is_collision else 0.0
            
            # Update aggregate statistics
            if is_collision:
                num_collisions += 1
                total_penetration += penetration
            
            # Track minimum distance
            if not is_collision and distance < min_distance:
                min_distance = distance
            
            # Store obstacle data
            obstacle_data.append(ObstacleProximity(
                obstacle_index=idx,
                distance=distance,
                is_collision=is_collision,
                penetration_depth=penetration
            ))
        
        # If no non-colliding obstacles, min_distance stays inf
        has_collision = is_out_of_bounds or num_collisions > 0
        
        return CollisionQueryResult(
            has_collision=has_collision,
            is_out_of_bounds=is_out_of_bounds,
            obstacle_data=obstacle_data,
            total_penetration_depth=total_penetration,
            num_collisions=num_collisions,
            min_obstacle_distance=min_distance
        )
    
    @staticmethod
    def box_to_box_distance(box1: Box3D, box2: Box3D) -> float:
        """
        Calculate the closest distance between two boxes.
        
        Uses FCL's distance computation which calculates the true minimum distance
        between the closest points on the two boxes (not just center-to-center).
        
        Args:
            box1: First Box3D object
            box2: Second Box3D object
            
        Returns:
            float: Closest distance between boxes (0 if colliding, negative if penetrating)
        """
        return box1.distance_to(box2)
    
    def get_num_obstacles(self) -> int:
        """Get the number of obstacles in the configuration space."""
        return len(self.obstacles)
    
    def get_obstacles(self) -> List[Box3D]:
        """Get a copy of the obstacles list."""
        return self.obstacles.copy()
    
    def reset(self):
        """Reset the configuration space to default state."""
        self.obstacles.clear()
        self.xMin = -10
        self.xMax = 10
        self.yMin = -10
        self.yMax = 10
        self.zMin = -10
        self.zMax = 10
        self._update_boundary_box()


# Test main function
if __name__ == "__main__":
    def test_configuration_space_3d():
        print("Testing ConfigurationSpace3D class...")
        
        # Create a configuration space
        config_space = ConfigurationSpace3D([0, 10, 0, 10, 0, 10])
        
        print(f"Configuration space bounds: x[{config_space.xMin}, {config_space.xMax}], "
              f"y[{config_space.yMin}, {config_space.yMax}], z[{config_space.zMin}, {config_space.zMax}]")
        
        # Create some obstacles
        obstacle1 = Box3D(
            center=PointXYZ(3, 3, 3),
            size=PointXYZ(1, 1, 1)
        )
        
        obstacle2 = Box3D(
            center=PointXYZ(7, 7, 5),
            size=PointXYZ(1.5, 1.5, 2)
        )
        
        # Add obstacles
        config_space.add_obstacle(obstacle1)
        config_space.add_obstacle(obstacle2)
        
        print(f"Added {config_space.get_num_obstacles()} obstacles")
        
        # Create a test vehicle box
        vehicle = Box3D(
            center=PointXYZ(5, 5, 5),
            size=PointXYZ(0.5, 0.5, 0.5)
        )
        
        # Check collision
        collision = config_space.check_collision(vehicle)
        print(f"Vehicle at (5, 5, 5) collision: {collision}")
        
        # Move vehicle closer to obstacle
        vehicle.update_transform(center=PointXYZ(3.5, 3.5, 3.5))
        collision = config_space.check_collision(vehicle)
        print(f"Vehicle at (3.5, 3.5, 3.5) collision: {collision}")
        
        # Get distance to nearest obstacle
        distance = config_space.get_nearest_obstacle_distance(vehicle)
        print(f"Distance to nearest obstacle: {distance:.3f}")
        
        # Test out of bounds
        vehicle.update_transform(center=PointXYZ(-1, 5, 5))
        collision = config_space.check_collision(vehicle)
        print(f"Vehicle at (-1, 5, 5) (out of bounds) collision: {collision}")
        
        # Test removing obstacles
        config_space.remove_obstacle_at_index(0)
        print(f"After removing first obstacle: {config_space.get_num_obstacles()} obstacles remain")
        
        # Test clearing all obstacles
        config_space.clear_obstacles()
        print(f"After clearing: {config_space.get_num_obstacles()} obstacles")
        
        print("Test completed!")
    
    # Run the test
    test_configuration_space_3d()
