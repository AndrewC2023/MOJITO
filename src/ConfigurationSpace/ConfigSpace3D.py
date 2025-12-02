import numpy as np
from typing import List, Dict, NamedTuple
import sys
from pathlib import Path
import fcl

# Add parent directory to path to allow imports
sys.path.append(str(Path(__file__).parent.parent))

from Utils.GeometryUtils import Box3D, PointXYZ
from ConfigurationSpace.Obstacles import Obstacle, StaticObstacle, SimpleMovingObstacle


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
    3D Configuration Space for motion planning with FCL-based obstacles.
    
    Manages a bounded 3D space with obstacles represented as Obstacle objects.
    Provides methods to add/remove obstacles and check for collisions using FCL.
    Methods accept both fcl.CollisionObject and Box3D (for backward compatibility).
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
        self.obstacles: List[Obstacle] = []
        
        # Create boundary box for the configuration space
        self._update_boundary_box()
    
    def _update_boundary_box(self):
        """Update the boundary box representing the field bounds."""
        center = np.array([
            (self.xMin + self.xMax) / 2,
            (self.yMin + self.yMax) / 2,
            (self.zMin + self.zMax) / 2
        ])
        size_x = self.xMax - self.xMin
        size_y = self.yMax - self.yMin
        size_z = self.zMax - self.zMin
        
        # Create FCL boundary collision object
        boundary_geom = fcl.Box(size_x, size_y, size_z)
        boundary_transform = fcl.Transform(np.eye(3), center)
        self.boundary_collision_object = fcl.CollisionObject(boundary_geom, boundary_transform)
    
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
    
    def add_obstacle(self, obstacle: Obstacle):
        """
        Add an Obstacle to the configuration space.
        
        Args:
            obstacle: Obstacle object (StaticObstacle, SimpleMovingObstacle, etc.) to add
        """
        if not isinstance(obstacle, Obstacle):
            raise ValueError("Obstacle must be an Obstacle object")
        self.obstacles.append(obstacle)
    
    def add_obstacles(self, obstacles: List[Obstacle]):
        """
        Add multiple Obstacles to the configuration space.
        
        Args:
            obstacles: List of Obstacle objects to add
        """
        for obstacle in obstacles:
            self.add_obstacle(obstacle)
    
    def remove_obstacle(self, obstacle: Obstacle):
        """
        Remove a specific obstacle from the configuration space.
        
        Args:
            obstacle: Obstacle object to remove
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
    
    @staticmethod
    def _to_collision_object(obj):
        """
        Convert input to fcl.CollisionObject.
        
        Args:
            obj: Either fcl.CollisionObject or Box3D object
            
        Returns:
            fcl.CollisionObject
            
        Raises:
            TypeError: If obj is neither fcl.CollisionObject nor Box3D
        """
        if isinstance(obj, fcl.CollisionObject):
            return obj
        elif isinstance(obj, Box3D):
            return obj.collision_object
        else:
            raise TypeError(f"Expected fcl.CollisionObject or Box3D, got {type(obj)}")
    
    def is_point_in_bounds(self, point) -> bool:
        """
        Check if a point is within the configuration space bounds.
        
        Args:
            point: PointXYZ, tuple, list, or numpy array [x, y, z]
            
        Returns:
            bool: True if point is within bounds, False otherwise
        """
        if isinstance(point, PointXYZ):
            x, y, z = point.x, point.y, point.z
        else:
            x, y, z = point[0], point[1], point[2]
        
        return (self.xMin <= x <= self.xMax and
                self.yMin <= y <= self.yMax and
                self.zMin <= z <= self.zMax)
    
    def is_in_bounds(self, obj) -> bool:
        """
        Check if an object is entirely within the configuration space bounds.
        
        Uses FCL to check if the object is fully contained within the boundary box.
        We create 6 boundary planes (walls) and check that the object doesn't
        collide with the "outside" of any wall.
        
        Args:
            obj: fcl.CollisionObject or Box3D object to check
            
        Returns:
            bool: True if object is entirely within bounds, False otherwise
        """
        collision_obj = self._to_collision_object(obj)
        
        # Check collision with each boundary wall from the outside
        # If we collide with a wall from outside, we're out of bounds
        request = fcl.CollisionRequest()
        result = fcl.CollisionResult()
        
        # Get object position to determine which walls to check
        transform = collision_obj.getTransform()
        pos = transform.getTranslation()
        cx, cy, cz = pos[0], pos[1], pos[2]
        
        # Simple and efficient: check if center point is in bounds
        # Then use distance to boundaries to verify object doesn't extend outside
        if not (self.xMin <= cx <= self.xMax and
                self.yMin <= cy <= self.yMax and
                self.zMin <= cz <= self.zMax):
            return False
        
        # Use distance computation to check clearance from each boundary
        # Create thin box planes for each boundary face
        epsilon = 0.01  # Thin plane thickness
        
        # Check each boundary face
        boundary_faces = [
            # X min face (left wall)
            (fcl.Box(epsilon, self.yMax - self.yMin, self.zMax - self.zMin),
             fcl.Transform(np.eye(3), [self.xMin - epsilon/2, (self.yMin + self.yMax)/2, (self.zMin + self.zMax)/2])),
            # X max face (right wall)
            (fcl.Box(epsilon, self.yMax - self.yMin, self.zMax - self.zMin),
             fcl.Transform(np.eye(3), [self.xMax + epsilon/2, (self.yMin + self.yMax)/2, (self.zMin + self.zMax)/2])),
            # Y min face (back wall)
            (fcl.Box(self.xMax - self.xMin, epsilon, self.zMax - self.zMin),
             fcl.Transform(np.eye(3), [(self.xMin + self.xMax)/2, self.yMin - epsilon/2, (self.zMin + self.zMax)/2])),
            # Y max face (front wall)
            (fcl.Box(self.xMax - self.xMin, epsilon, self.zMax - self.zMin),
             fcl.Transform(np.eye(3), [(self.xMin + self.xMax)/2, self.yMax + epsilon/2, (self.zMin + self.zMax)/2])),
            # Z min face (bottom wall)
            (fcl.Box(self.xMax - self.xMin, self.yMax - self.yMin, epsilon),
             fcl.Transform(np.eye(3), [(self.xMin + self.xMax)/2, (self.yMin + self.yMax)/2, self.zMin - epsilon/2])),
            # Z max face (top wall)
            (fcl.Box(self.xMax - self.xMin, self.yMax - self.yMin, epsilon),
             fcl.Transform(np.eye(3), [(self.xMin + self.xMax)/2, (self.yMin + self.yMax)/2, self.zMax + epsilon/2])),
        ]
        
        # Check if object collides with any external boundary face
        for geom, transform in boundary_faces:
            wall = fcl.CollisionObject(geom, transform)
            result = fcl.CollisionResult()  # Create fresh result for each check
            ret = fcl.collide(collision_obj, wall, request, result)
            if ret > 0:  # Collision with boundary wall means out of bounds
                return False
        
        return True
    
    def check_collision(self, obj, t: float = 0.0) -> bool:
        """
        Check if an object collides with field boundaries or any obstacles at time t.
        
        Args:
            obj: fcl.CollisionObject or Box3D object to check for collision
            t: Time at which to check collision (default: 0.0)
            
        Returns:
            bool: True if collision detected, False otherwise
        """
        collision_obj = self._to_collision_object(obj)
        
        # Check if object is out of bounds
        if not self.is_in_bounds(obj):
            return True
        
        # Early exit if no obstacles
        if not self.obstacles:
            return False
        
        # Check collision with all obstacles at time t
        request = fcl.CollisionRequest()
        result = fcl.CollisionResult()
        
        for obstacle in self.obstacles:
            obstacle_collision_obj = obstacle.get_collision_object(t)
            ret = fcl.collide(collision_obj, obstacle_collision_obj, request, result)
            if ret > 0:  # Collision detected
                return True
        
        return False
    
    def check_collision_with_obstacles_only(self, obj, t: float = 0.0) -> bool:
        """
        Check if an object collides with any obstacles at time t (ignores boundaries).
        
        Args:
            obj: fcl.CollisionObject or Box3D object to check for collision
            t: Time at which to check collision (default: 0.0)
            
        Returns:
            bool: True if collision with obstacle detected, False otherwise
        """
        collision_obj = self._to_collision_object(obj)
        
        request = fcl.CollisionRequest()
        result = fcl.CollisionResult()
        
        for obstacle in self.obstacles:
            obstacle_collision_obj = obstacle.get_collision_object(t)
            ret = fcl.collide(collision_obj, obstacle_collision_obj, request, result)
            if ret > 0:  # Collision detected
                return True
        return False
    
    def get_nearest_obstacle_distance(self, obj, t: float = 0.0) -> float:
        """
        Get the distance to the nearest obstacle at time t.
        
        Args:
            obj: fcl.CollisionObject or Box3D object to check distance from
            t: Time at which to check distance (default: 0.0)
            
        Returns:
            float: Minimum distance to nearest obstacle (inf if no obstacles)
        """
        if not self.obstacles:
            return float('inf')
        
        collision_obj = self._to_collision_object(obj)
        
        min_distance = float('inf')
        request = fcl.DistanceRequest()
        result = fcl.DistanceResult()
        
        for obstacle in self.obstacles:
            obstacle_collision_obj = obstacle.get_collision_object(t)
            fcl.distance(collision_obj, obstacle_collision_obj, request, result)
            distance = result.min_distance
            
            # Early exit if already in collision
            if distance <= 0:
                return 0.0
            if distance < min_distance:
                min_distance = distance
        
        return min_distance
    
    def get_distance_to_boundaries(self, obj) -> float:
        """
        Get the distance from an object to the nearest boundary wall.
        
        Uses FCL distance computation against each boundary face.
        
        Args:
            obj: fcl.CollisionObject or Box3D object to check distance from
            
        Returns:
            float: Minimum distance to nearest boundary (negative if penetrating/outside bounds)
        """
        collision_obj = self._to_collision_object(obj)
        
        # Create thin box planes for each boundary face
        epsilon = 0.01  # Thin plane thickness
        
        boundary_faces = [
            # X min face (left wall)
            (fcl.Box(epsilon, self.yMax - self.yMin, self.zMax - self.zMin),
             fcl.Transform(np.eye(3), [self.xMin - epsilon/2, (self.yMin + self.yMax)/2, (self.zMin + self.zMax)/2])),
            # X max face (right wall)
            (fcl.Box(epsilon, self.yMax - self.yMin, self.zMax - self.zMin),
             fcl.Transform(np.eye(3), [self.xMax + epsilon/2, (self.yMin + self.yMax)/2, (self.zMin + self.zMax)/2])),
            # Y min face (back wall)
            (fcl.Box(self.xMax - self.xMin, epsilon, self.zMax - self.zMin),
             fcl.Transform(np.eye(3), [(self.xMin + self.xMax)/2, self.yMin - epsilon/2, (self.zMin + self.zMax)/2])),
            # Y max face (front wall)
            (fcl.Box(self.xMax - self.xMin, epsilon, self.zMax - self.zMin),
             fcl.Transform(np.eye(3), [(self.xMin + self.xMax)/2, self.yMax + epsilon/2, (self.zMin + self.zMax)/2])),
            # Z min face (bottom wall)
            (fcl.Box(self.xMax - self.xMin, self.yMax - self.yMin, epsilon),
             fcl.Transform(np.eye(3), [(self.xMin + self.xMax)/2, (self.yMin + self.yMax)/2, self.zMin - epsilon/2])),
            # Z max face (top wall)
            (fcl.Box(self.xMax - self.xMin, self.yMax - self.yMin, epsilon),
             fcl.Transform(np.eye(3), [(self.xMin + self.xMax)/2, (self.yMin + self.yMax)/2, self.zMax + epsilon/2])),
        ]
        
        # Find minimum distance to any boundary face
        min_distance = float('inf')
        request = fcl.DistanceRequest()
        result = fcl.DistanceResult()
        
        for geom, transform in boundary_faces:
            wall = fcl.CollisionObject(geom, transform)
            fcl.distance(collision_obj, wall, request, result)
            distance = result.min_distance
            
            if distance < min_distance:
                min_distance = distance
        
        # Subtract epsilon to account for the thin plane thickness
        return min_distance - epsilon/2
    
    def query_collision_detailed(self, obj, t: float = 0.0) -> CollisionQueryResult:
        """
        Perform comprehensive collision query with detailed information for cost functions at time t.
        
        Returns information about all obstacles including:
        - Which obstacles are being collided with
        - Penetration depth for each collision
        - Distance to each obstacle (even non-colliding ones)
        - Total penetration depth (sum of all collisions)
        
        This provides smooth, continuous data for optimization cost functions.
        
        Args:
            obj: fcl.CollisionObject or Box3D object to query
            t: Time at which to check collision (default: 0.0)
            
        Returns:
            CollisionQueryResult with complete proximity information
        """
        collision_obj = self._to_collision_object(obj)
        
        # Check boundary collision
        is_out_of_bounds = not self.is_in_bounds(obj)
        
        # Collect data for each obstacle
        obstacle_data = []
        total_penetration = 0.0
        num_collisions = 0
        min_distance = float('inf')
        
        dist_request = fcl.DistanceRequest()
        dist_result = fcl.DistanceResult()
        
        for idx, obstacle in enumerate(self.obstacles):
            obstacle_collision_obj = obstacle.get_collision_object(t)
            
            # Get distance (negative if penetrating)
            fcl.distance(collision_obj, obstacle_collision_obj, dist_request, dist_result)
            distance = dist_result.min_distance
            
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
    def compute_distance(obj1, obj2) -> float:
        """
        Calculate the closest distance between two objects.
        
        Uses FCL's distance computation which calculates the true minimum distance
        between the closest points on the two objects.
        
        Args:
            obj1: First fcl.CollisionObject or Box3D object
            obj2: Second fcl.CollisionObject or Box3D object
            
        Returns:
            float: Closest distance between objects (0 if colliding, negative if penetrating)
        """
        collision_obj1 = ConfigurationSpace3D._to_collision_object(obj1)
        collision_obj2 = ConfigurationSpace3D._to_collision_object(obj2)
        
        request = fcl.DistanceRequest()
        result = fcl.DistanceResult()
        fcl.distance(collision_obj1, collision_obj2, request, result)
        
        return result.min_distance
    
    def get_num_obstacles(self) -> int:
        """Get the number of obstacles in the configuration space."""
        return len(self.obstacles)
    
    def get_obstacles(self) -> List[Obstacle]:
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

    print("Testing ConfigurationSpace3D class...")
    
    # Create a configuration space
    config_space = ConfigurationSpace3D([0, 10, 0, 10, 0, 10])
    
    print(f"Configuration space bounds: x[{config_space.xMin}, {config_space.xMax}], "
            f"y[{config_space.yMin}, {config_space.yMax}], z[{config_space.zMin}, {config_space.zMax}]")
    
    building_geometry = fcl.Box(2.0, 2.0, 3.0)  # 2x2x3 meter building
    building_transform = fcl.Transform(np.eye(3), [3.0, 3.0, 1.5])
    building = StaticObstacle(building_geometry, building_transform)
    config_space.add_obstacle(building)
    
    print(f"Added {config_space.get_num_obstacles()} obstacles")
    
    # Create a test vehicle using FCL directly
    vehicle_geom = fcl.Box(0.5, 0.5, 0.5)
    vehicle_transform = fcl.Transform(np.eye(3), [5.0, 5.0, 5.0])
    vehicle = fcl.CollisionObject(vehicle_geom, vehicle_transform)
    
    # Check collision
    collision = config_space.check_collision(vehicle)
    print(f"Vehicle at (5, 5, 5) collision: {collision}")
    
    # Move vehicle closer to obstacle
    vehicle_transform = fcl.Transform(np.eye(3), [3.5, 3.5, 3.5])
    vehicle.setTransform(vehicle_transform)
    collision = config_space.check_collision(vehicle)
    print(f"Vehicle at (3.5, 3.5, 3.5) collision: {collision}")
    
    # Get distance to nearest obstacle
    distance = config_space.get_nearest_obstacle_distance(vehicle)
    print(f"Distance to nearest obstacle: {distance:.3f}")
    
    # Test out of bounds
    vehicle_transform = fcl.Transform(np.eye(3), [-1.0, 5.0, 5.0])
    vehicle.setTransform(vehicle_transform)
    collision = config_space.check_collision(vehicle)
    print(f"Vehicle at (-1, 5, 5) (out of bounds) collision: {collision}")
    
    # Test removing obstacles
    config_space.remove_obstacle_at_index(0)
    print(f"After removing first obstacle: {config_space.get_num_obstacles()} obstacles remain")
    
    # Test clearing all obstacles
    config_space.clear_obstacles()
    print(f"After clearing: {config_space.get_num_obstacles()} obstacles")
    
    print("Test completed!")
    
