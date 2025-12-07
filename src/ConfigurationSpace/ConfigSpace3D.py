import numpy as np
from typing import List, Dict, NamedTuple, Optional, Tuple, Union
import sys
from pathlib import Path
import fcl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches

# Add parent directory to path to allow imports
sys.path.append(str(Path(__file__).parent.parent))

from Utils.GeometryUtils import PointXYZ
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
    
    This structure provides all information needed for smoother cost functions
    in optimization. It allows you to:
    
    1. Heavily vary collision penalties using total_penetration_depth
       - Accounts for obstacle size: grazing a pebble vs hitting a wall
       Or just hitting a corner versus being inside a wall, or hitting multiple obstacles

    2. Create smooth proximity costs to encourage safe distances
       - Access distance to each obstacle even when not colliding
    
    3. Handle boundary violations with is_out_of_bounds flag
    
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
    All methods accept fcl.CollisionObject for collision and distance queries.

    Based heavily on the 2D ConfigurationSpace2D class, but extended to 3D and using the
    free collision library for geometry checks ratehr than functions in GeometryUtils.
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
                raise ValueError(f"Dimensions must have at least 6 values, got {len(dimensions)}")
        
        # Validate that min < max for all dimensions
        self._validate_dimensions()
        
        # Initialize obstacle list
        self.obstacles: List[Obstacle] = []
        
        # Create boundary faces for the configuration space
        self._create_boundary_faces()
    
    def _validate_dimensions(self):
        """Validate that min < max for all dimensions.
        
        Raises:
            ValueError: If any min value is >= its corresponding max value
        """
        if self.xMin >= self.xMax:
            raise ValueError(f"xMin ({self.xMin}) must be less than xMax ({self.xMax})")
        if self.yMin >= self.yMax:
            raise ValueError(f"yMin ({self.yMin}) must be less than yMax ({self.yMax})")
        if self.zMin >= self.zMax:
            raise ValueError(f"zMin ({self.zMin}) must be less than zMax ({self.zMax})")
    
    def _create_boundary_faces(self):
        """Create thin boundary face collision objects for efficient boundary checking.
        
        Creates 6 thin box planes positioned just outside the bounds. These are reused
        for all boundary checks instead of being reconstructed each time.
        """
        epsilon = 0.01  # Thin plane thickness
        
        # Create collision objects for each boundary face
        self._boundary_faces = [
            # X min face (left wall)
            fcl.CollisionObject(
                fcl.Box(epsilon, self.yMax - self.yMin, self.zMax - self.zMin),
                fcl.Transform(np.eye(3), [self.xMin - epsilon/2, (self.yMin + self.yMax)/2, (self.zMin + self.zMax)/2])
            ),
            # X max face (right wall)
            fcl.CollisionObject(
                fcl.Box(epsilon, self.yMax - self.yMin, self.zMax - self.zMin),
                fcl.Transform(np.eye(3), [self.xMax + epsilon/2, (self.yMin + self.yMax)/2, (self.zMin + self.zMax)/2])
            ),
            # Y min face (back wall)
            fcl.CollisionObject(
                fcl.Box(self.xMax - self.xMin, epsilon, self.zMax - self.zMin),
                fcl.Transform(np.eye(3), [(self.xMin + self.xMax)/2, self.yMin - epsilon/2, (self.zMin + self.zMax)/2])
            ),
            # Y max face (front wall)
            fcl.CollisionObject(
                fcl.Box(self.xMax - self.xMin, epsilon, self.zMax - self.zMin),
                fcl.Transform(np.eye(3), [(self.xMin + self.xMax)/2, self.yMax + epsilon/2, (self.zMin + self.zMax)/2])
            ),
            # Z min face (bottom wall)
            fcl.CollisionObject(
                fcl.Box(self.xMax - self.xMin, self.yMax - self.yMin, epsilon),
                fcl.Transform(np.eye(3), [(self.xMin + self.xMax)/2, (self.yMin + self.yMax)/2, self.zMin - epsilon/2])
            ),
            # Z max face (top wall)
            fcl.CollisionObject(
                fcl.Box(self.xMax - self.xMin, self.yMax - self.yMin, epsilon),
                fcl.Transform(np.eye(3), [(self.xMin + self.xMax)/2, (self.yMin + self.yMax)/2, self.zMax + epsilon/2])
            ),
        ]
    
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
        self._create_boundary_faces()
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
        First checks if center is in bounds (fast rejection), then uses FCL collision
        with boundary faces to verify object doesn't extend outside.
        
        Args:
            obj: fcl.CollisionObject to check
            
        Returns:
            bool: True if object is entirely within bounds, False otherwise
        """
        # Quick check: is center point in bounds
        transform = obj.getTransform()
        pos = transform.getTranslation()
        cx, cy, cz = pos[0], pos[1], pos[2]
        
        # If center is out of bounds, object is definitely out of bounds
        if not (self.xMin <= cx <= self.xMax and
                self.yMin <= cy <= self.yMax and
                self.zMin <= cz <= self.zMax):
            return False
        
        # Center is in bounds, now check if object extends outside boundaries
        # Reuse single request object for all checks (small optimization)
        request = fcl.CollisionRequest()
        
        for wall in self._boundary_faces:
            result = fcl.CollisionResult()
            ret = fcl.collide(obj, wall, request, result)
            if ret > 0:  # Collision with boundary wall means out of bounds
                return False
        
        return True
    
    def check_collision(self, obj, t: float = 0.0) -> bool:
        """
        Check if an object collides with field boundaries or any obstacles at time t.
        
        Args:
            obj: fcl.CollisionObject to check for collision
            t: Time at which to check collision (default: 0.0)
            
        Returns:
            bool: True if collision detected, False otherwise
        """
        # Check if object is out of bounds
        if not self.is_in_bounds(obj):
            return True
        
        # Early exit if no obstacles
        if not self.obstacles:
            return False
        
        # Check collision with all obstacles at time t
        # Reuse request and result objects for efficiency
        request = fcl.CollisionRequest(enable_contact=False)  # Disable contact computation for speed
        
        for obstacle in self.obstacles:
            result = fcl.CollisionResult()
            obstacle_collision_obj = obstacle.get_collision_object(t)
            ret = fcl.collide(obj, obstacle_collision_obj, request, result)
            if ret > 0:  # Collision detected
                return True
        
        return False
    
    def check_collision_with_obstacles_only(self, obj, t: float = 0.0) -> bool:
        """
        Check if an object collides with any obstacles at time t (ignores boundaries).
        
        Args:
            obj: fcl.CollisionObject to check for collision
            t: Time at which to check collision (default: 0.0)
            
        Returns:
            bool: True if collision with obstacle detected, False otherwise
        """
        # Early exit if no obstacles
        if not self.obstacles:
            return False
        
        # Reuse request and result objects for efficiency
        request = fcl.CollisionRequest(enable_contact=False)  # Disable contact computation for speed
        
        for obstacle in self.obstacles:
            result = fcl.CollisionResult()
            obstacle_collision_obj = obstacle.get_collision_object(t)
            ret = fcl.collide(obj, obstacle_collision_obj, request, result)
            if ret > 0:  # Collision detected
                return True
        return False
    
    def get_nearest_obstacle_distance(self, obj, t: float = 0.0) -> float:
        """
        Get the distance to the nearest obstacle at time t.
        
        Args:
            obj: fcl.CollisionObject to check distance from
            t: Time at which to check distance (default: 0.0)
            
        Returns:
            float: Minimum distance to nearest obstacle (inf if no obstacles)
        """
        if not self.obstacles:
            return float('inf')
        
        min_distance = float('inf')
        # Reuse request and result objects for efficiency
        request = fcl.DistanceRequest(enable_nearest_points=False)  # Disable nearest points for speed
        
        for obstacle in self.obstacles:
            result = fcl.DistanceResult()
            obstacle_collision_obj = obstacle.get_collision_object(t)
            fcl.distance(obj, obstacle_collision_obj, request, result)
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
            obj: fcl.CollisionObject to check distance from
            
        Returns:
            float: Minimum distance to nearest boundary (negative if penetrating/outside bounds)
        """
        # Find minimum distance to any boundary face (using cached faces)
        min_distance = float('inf')
        # Reuse request and result objects for efficiency
        request = fcl.DistanceRequest()
        
        epsilon = 0.01  # Thin plane thickness used in boundary face creation
        
        for wall in self._boundary_faces:
            result = fcl.DistanceResult()
            fcl.distance(obj, wall, request, result)
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
            obj: fcl.CollisionObject to query
            t: Time at which to check collision (default: 0.0)
            
        Returns:
            CollisionQueryResult with complete proximity information
        """
        # Check boundary collision
        is_out_of_bounds = not self.is_in_bounds(obj)
        
        # Collect data for each obstacle
        obstacle_data = []
        total_penetration = 0.0
        num_collisions = 0
        min_distance = float('inf')
        
        # Reuse request and result objects for efficiency
        # Enable nearest points is disabled by default for speed
        dist_request = fcl.DistanceRequest(enable_nearest_points=False)
        
        for idx, obstacle in enumerate(self.obstacles):
            dist_result = fcl.DistanceResult()
            obstacle_collision_obj = obstacle.get_collision_object(t)
            
            # Get distance (negative if penetrating)
            fcl.distance(obj, obstacle_collision_obj, dist_request, dist_result)
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
            obj1: First fcl.CollisionObject
            obj2: Second fcl.CollisionObject
            
        Returns:
            float: Closest distance between objects (0 if colliding, negative if penetrating)
        """
        # Disable nearest points computation for speed (we only need distance)
        request = fcl.DistanceRequest(enable_nearest_points=False)
        result = fcl.DistanceResult()
        fcl.distance(obj1, obj2, request, result)
        
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
        self._create_boundary_faces()
    
    @staticmethod
    def _get_box_vertices(center: np.ndarray, rotation: np.ndarray, 
                          size: np.ndarray) -> np.ndarray:
        """
        Get the 8 vertices of a 3D box given its center, rotation, and size.
        
        Args:
            center: 3D center position [x, y, z]
            rotation: 3x3 rotation matrix
            size: Box dimensions [width, depth, height]
            
        Returns:
            8x3 array of vertices
        """
        # Define box vertices in local frame (centered at origin)
        half_size = size / 2.0
        local_vertices = np.array([
            [-half_size[0], -half_size[1], -half_size[2]],
            [ half_size[0], -half_size[1], -half_size[2]],
            [ half_size[0],  half_size[1], -half_size[2]],
            [-half_size[0],  half_size[1], -half_size[2]],
            [-half_size[0], -half_size[1],  half_size[2]],
            [ half_size[0], -half_size[1],  half_size[2]],
            [ half_size[0],  half_size[1],  half_size[2]],
            [-half_size[0],  half_size[1],  half_size[2]],
        ])
        
        # Transform vertices to world frame
        world_vertices = (rotation @ local_vertices.T).T + center
        return world_vertices
    
    @staticmethod
    def _get_box_faces(vertices: np.ndarray) -> List[np.ndarray]:
        """
        Get the 6 faces of a box from its 8 vertices.
        
        Args:
            vertices: 8x3 array of vertices
            
        Returns:
            List of 6 face arrays, each 4x3 (quad vertices)
        """
        # Define faces by vertex indices
        face_indices = [
            [0, 1, 2, 3],  # Bottom
            [4, 5, 6, 7],  # Top
            [0, 1, 5, 4],  # Front
            [2, 3, 7, 6],  # Back
            [0, 3, 7, 4],  # Left
            [1, 2, 6, 5],  # Right
        ]
        
        faces = [vertices[face_idx] for face_idx in face_indices]
        return faces
    
    def plot_configuration_space(self, 
                                  t: float = 0.0,
                                  ax: Optional[plt.Axes] = None,
                                  show_bounds: bool = True,
                                  show_obstacles: bool = True,
                                  obstacle_alpha: float = 0.3,
                                  obstacle_color: str = 'red',
                                  bounds_alpha: float = 0.1,
                                  bounds_color: str = 'gray',
                                  bounds_edge_color: str = 'black',
                                  bounds_edge_width: float = 2.0,
                                  title: Optional[str] = None) -> plt.Axes:
        """
        Plot the 3D configuration space with boundaries and obstacles.
        
        Visualizes the bounded region and all obstacles (static and dynamic) at a given time.
        Obstacles are rendered semi-transparent to allow viewing of overlapping trajectories.
        
        Args:
            t: Time at which to evaluate dynamic obstacles (default: 0.0)
            ax: Matplotlib 3D axes. If None, creates new figure (default: None)
            show_bounds: Whether to show the configuration space boundaries (default: True)
            show_obstacles: Whether to show obstacles (default: True)
            obstacle_alpha: Transparency of obstacles, 0=transparent, 1=opaque (default: 0.3)
            obstacle_color: Color for obstacle rendering (default: 'red')
            bounds_alpha: Transparency of boundary box (default: 0.1)
            bounds_color: Color for boundary faces (default: 'gray')
            bounds_edge_color: Color for boundary edges (default: 'black')
            bounds_edge_width: Width of boundary edges (default: 2.0)
            title: Plot title. If None, generates default title (default: None)
            
        Returns:
            Matplotlib 3D axes with the plot
        """
        # Create axes if not provided
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
        
        # Plot boundary box
        if show_bounds:
            center = np.array([
                (self.xMin + self.xMax) / 2,
                (self.yMin + self.yMax) / 2,
                (self.zMin + self.zMax) / 2
            ])
            size = np.array([
                self.xMax - self.xMin,
                self.yMax - self.yMin,
                self.zMax - self.zMin
            ])
            
            vertices = self._get_box_vertices(center, np.eye(3), size)
            faces = self._get_box_faces(vertices)
            
            # Add semi-transparent boundary faces
            poly = Poly3DCollection(faces, alpha=bounds_alpha, 
                                   facecolor=bounds_color, 
                                   edgecolor=bounds_edge_color,
                                   linewidth=bounds_edge_width)
            ax.add_collection3d(poly)
        
        # Plot obstacles
        if show_obstacles:
            for obstacle in self.obstacles:
                transform = obstacle.get_transform(t)
                center = transform.getTranslation()
                rotation = transform.getRotation()
                
                # Extract geometry dimensions based on type
                geom = obstacle.geometry


                # handle the fcl geometries we support
                if isinstance(geom, fcl.Box):
                    # Box obstacle
                    size = geom.side
                    vertices = self._get_box_vertices(center, rotation, size)
                    faces = self._get_box_faces(vertices)
                    
                    poly = Poly3DCollection(faces, alpha=obstacle_alpha,
                                          facecolor=obstacle_color,
                                          edgecolor='darkred',
                                          linewidth=1.0)
                    ax.add_collection3d(poly)
                
                elif isinstance(geom, fcl.Sphere):
                    # Sphere obstacle - use wireframe
                    radius = geom.radius
                    u = np.linspace(0, 2 * np.pi, 20)
                    v = np.linspace(0, np.pi, 20)
                    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
                    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
                    z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
                    ax.plot_surface(x, y, z, alpha=obstacle_alpha, color=obstacle_color)
                
                elif isinstance(geom, fcl.Cylinder):
                    # Cylinder obstacle - approximate with polygons
                    radius = geom.radius
                    height = geom.lz
                    
                    # Create cylinder vertices
                    theta = np.linspace(0, 2*np.pi, 20)
                    z_cyl = np.array([-height/2, height/2])
                    
                    # Generate cylinder surface in local frame
                    for i in range(len(theta)-1):
                        # Four corners of the rectangular face
                        local_face = np.array([
                            [radius*np.cos(theta[i]), radius*np.sin(theta[i]), -height/2],
                            [radius*np.cos(theta[i+1]), radius*np.sin(theta[i+1]), -height/2],
                            [radius*np.cos(theta[i+1]), radius*np.sin(theta[i+1]), height/2],
                            [radius*np.cos(theta[i]), radius*np.sin(theta[i]), height/2],
                        ])
                        
                        # Transform to world frame
                        world_face = (rotation @ local_face.T).T + center
                        
                        poly = Poly3DCollection([world_face], alpha=obstacle_alpha,
                                              facecolor=obstacle_color,
                                              edgecolor='darkred',
                                              linewidth=0.5)
                        ax.add_collection3d(poly)
                    
                    # Add top and bottom caps
                    for z_val in [-height/2, height/2]:
                        cap_vertices = []
                        for th in theta:
                            local_pt = np.array([radius*np.cos(th), radius*np.sin(th), z_val])
                            world_pt = rotation @ local_pt + center
                            cap_vertices.append(world_pt)
                        
                        poly = Poly3DCollection([cap_vertices], alpha=obstacle_alpha,
                                              facecolor=obstacle_color,
                                              edgecolor='darkred',
                                              linewidth=0.5)
                        ax.add_collection3d(poly)
        
        # Set axis limits
        ax.set_xlim([self.xMin, self.xMax])
        ax.set_ylim([self.yMin, self.yMax])
        ax.set_zlim([self.zMin, self.zMax])
        
        # Labels
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        
        # Title
        if title is None:
            title = f'Configuration Space (t={t:.2f}s)'
        ax.set_title(title)
        
        # Add legend
        legend_elements = []
        if show_obstacles:
            legend_elements.append(mpatches.Patch(facecolor=obstacle_color, 
                                                 alpha=obstacle_alpha,
                                                 edgecolor='darkred',
                                                 label='Obstacles'))
        if show_bounds:
            legend_elements.append(mpatches.Patch(facecolor=bounds_color,
                                                 alpha=bounds_alpha,
                                                 edgecolor=bounds_edge_color,
                                                 label='Bounds'))
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right')
        
        return ax
    
    def plot_vehicle_trajectory(self,
                                trajectory: Union[np.ndarray, List[np.ndarray]],
                                times: Optional[List[float]] = None,
                                vehicle_geometry: Optional[fcl.CollisionGeometry] = None,
                                sample_indices: Optional[List[int]] = None,
                                ax: Optional[plt.Axes] = None,
                                trajectory_color: str = 'blue',
                                trajectory_width: float = 2.0,
                                vehicle_alpha: float = 0.5,
                                vehicle_color: str = 'green',
                                show_trajectory_line: bool = True,
                                show_vehicle_boxes: bool = True,
                                t_obstacles: float = 0.0,
                                **config_space_kwargs) -> plt.Axes:
        """
        Plot vehicle trajectory through configuration space with bounding boxes at sample points.

        Args:
            trajectory: Array of positions, shape (N, 3) or list of N position arrays [x, y, z]
            times: Optional list of times corresponding to trajectory points.
                Used with dynamic obstacles if t_obstacles='from_trajectory'
            vehicle_geometry: FCL geometry for the vehicle (e.g., fcl.Box, fcl.Sphere).
                Required if show_vehicle_boxes=True
            sample_indices: Indices at which to draw vehicle bounding boxes.
                If None, draws at evenly spaced intervals (default: 10 samples)
            ax: Matplotlib 3D axes. If None, creates new figure
            trajectory_color: Color for trajectory line (default: 'blue')
            trajectory_width: Width of trajectory line (default: 2.0)
            vehicle_alpha: Transparency of vehicle boxes (default: 0.5)
            vehicle_color: Color for vehicle boxes (default: 'green')
            show_trajectory_line: Whether to show the trajectory line (default: True)
            show_vehicle_boxes: Whether to show vehicle bounding boxes (default: True)
            t_obstacles: Time at which to show obstacles (default: 0.0)
            **config_space_kwargs: Additional arguments passed to plot_configuration_space
                                  (e.g., obstacle_alpha, bounds_alpha, etc.)
            
        Returns:
            Matplotlib 3D axes with the plot
        """
        # Convert trajectory to numpy array
        if isinstance(trajectory, list):
            trajectory = np.array(trajectory)
        
        if trajectory.shape[1] != 3:
            raise ValueError(f"Trajectory must have shape (N, 3), got {trajectory.shape}")
        
        # Plot configuration space first
        if ax is None:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
        
        ax = self.plot_configuration_space(t=t_obstacles, ax=ax, **config_space_kwargs)
        
        # Plot trajectory line
        if show_trajectory_line:
            ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                   color=trajectory_color, linewidth=trajectory_width, 
                   label='Trajectory', zorder=10)
        
        # Plot vehicle boxes at sample points
        if show_vehicle_boxes:
            if vehicle_geometry is None:
                raise ValueError("vehicle_geometry required when show_vehicle_boxes=True")
            
            # Determine sample indices
            if sample_indices is None:
                # Default: 10 evenly spaced samples
                n_samples = min(10, len(trajectory))
                sample_indices = np.linspace(0, len(trajectory)-1, n_samples, dtype=int)
            
            for idx in sample_indices:
                position = trajectory[idx]
                
                # Extract geometry dimensions
                if isinstance(vehicle_geometry, fcl.Box):
                    size = vehicle_geometry.side
                    vertices = self._get_box_vertices(position, np.eye(3), size)
                    faces = self._get_box_faces(vertices)
                    
                    poly = Poly3DCollection(faces, alpha=vehicle_alpha,
                                          facecolor=vehicle_color,
                                          edgecolor='darkgreen',
                                          linewidth=1.5,
                                          zorder=5)
                    ax.add_collection3d(poly)
                
                elif isinstance(vehicle_geometry, fcl.Sphere):
                    radius = vehicle_geometry.radius
                    a = np.linspace(0, 2 * np.pi, 15)
                    b = np.linspace(0, np.pi, 15)
                    x = radius * np.outer(np.cos(a), np.sin(b)) + position[0]
                    y = radius * np.outer(np.sin(a), np.sin(b)) + position[1]
                    z = radius * np.outer(np.ones(np.size(a)), np.cos(b)) + position[2]
                    ax.plot_surface(x, y, z, alpha=vehicle_alpha, color=vehicle_color, zorder=5)
        
        # Update legend
        if show_vehicle_boxes:
            vehicle_patch = mpatches.Patch(facecolor=vehicle_color, alpha=vehicle_alpha,
                                          edgecolor='darkgreen', label='Vehicle')
            handles, labels = ax.get_legend_handles_labels()
            handles.append(vehicle_patch)
            ax.legend(handles=handles, loc='upper right')
        
        return ax


# Test main function (maybe depreciated?)
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
    
