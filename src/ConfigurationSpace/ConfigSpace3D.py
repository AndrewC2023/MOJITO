import numpy as np
from typing import List
import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.append(str(Path(__file__).parent.parent))

from Utils.GeometryUtils import Box3D, PointXYZ

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
        # Get the vertices of the box
        vertices = box.get_vertices()
        
        # Check if all vertices are within bounds
        for vertex in vertices:
            if not self.is_point_in_bounds(vertex):
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
        
        # Check collision with all obstacles
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
            if distance < min_distance:
                min_distance = distance
        
        return min_distance
    
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
