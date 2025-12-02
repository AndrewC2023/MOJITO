import numpy as np
import fcl
from typing import List, Tuple, Optional
from abc import ABC, abstractmethod
import warnings


class Obstacle(ABC):
    """
    Base class for obstacles in the configuration space.
    
    All obstacles must provide collision checking functionality at a given time.
    """
    
    def __init__(self, geometry: fcl.CollisionGeometry):
        """
        Initialize an obstacle with FCL geometry.
        
        Args:
            geometry: FCL collision geometry (Box, Sphere, Cylinder, etc.)
        """
        self.geometry = geometry
    
    @abstractmethod
    def get_collision_object(self, t: float = 0.0) -> fcl.CollisionObject:
        """
        Get the FCL collision object at a specific time.
        
        Args:
            t: Time at which to query the obstacle (default: 0.0)
            
        Returns:
            fcl.CollisionObject ready for collision checking
        """
        pass
    
    @abstractmethod
    def get_transform(self, t: float = 0.0) -> fcl.Transform:
        """
        Get the transform of the obstacle at a specific time.
        
        Args:
            t: Time at which to query the obstacle transform (default: 0.0)
            
        Returns:
            fcl.Transform representing the obstacle's pose
        """
        pass


class StaticObstacle(Obstacle):
    """
    Static obstacle that doesn't move over time.
    
    Maintains a single transform and collision object that remains constant.
    """
    
    def __init__(self, geometry: fcl.CollisionGeometry, transform: fcl.Transform):
        """
        Initialize a static obstacle.
        
        Args:
            geometry: FCL collision geometry
            transform: FCL transform defining the obstacle's pose
        """
        super().__init__(geometry)
        self.transform = transform
        self.collision_object = fcl.CollisionObject(self.geometry, self.transform)
    
    def get_collision_object(self, t: float = 0.0) -> fcl.CollisionObject:
        """
        Get the collision object (time parameter ignored for static obstacles).
        
        Args:
            t: Time parameter (ignored for static obstacles)
            
        Returns:
            fcl.CollisionObject for collision checking
        """
        return self.collision_object
    
    def get_transform(self, t: float = 0.0) -> fcl.Transform:
        """
        Get the transform (time parameter ignored for static obstacles).
        
        Args:
            t: Time parameter (ignored for static obstacles)
            
        Returns:
            fcl.Transform of the obstacle
        """
        return self.transform


class SimpleMovingObstacle(Obstacle):
    """
    Simple moving obstacle defined by a list of transform-time pairs.
    
    Interpolates between transforms when queried at times between keyframes.
    If queried outside the time range, clamps to the nearest extent with a warning.
    """
    
    def __init__(self, geometry: fcl.CollisionGeometry, 
                 trajectory: List[Tuple[fcl.Transform, float]]):
        """
        Initialize a simple moving obstacle.
        
        Args:
            geometry: FCL collision geometry
            trajectory: List of (transform, time) tuples defining the obstacle's motion.
                       Must be sorted by time in ascending order.
                       
        Raises:
            ValueError: If trajectory is empty or not sorted by time
        """
        super().__init__(geometry)
        
        if not trajectory:
            raise ValueError("Trajectory must contain at least one (transform, time) pair")
        
        # Sort trajectory by time and validate
        self.trajectory = sorted(trajectory, key=lambda x: x[1])
        
        # Validate that times are unique
        times = [t for _, t in self.trajectory]
        if len(times) != len(set(times)):
            raise ValueError("Trajectory times must be unique")
        
        # Cache time bounds
        self.t_min = self.trajectory[0][1]
        self.t_max = self.trajectory[-1][1]
    
    def _clamp_time(self, t: float) -> float:
        """
        Clamp time to valid range and warn if out of bounds.
        
        Args:
            t: Time to clamp
            
        Returns:
            Clamped time within [t_min, t_max]
        """
        if t < self.t_min:
            warnings.warn(f"Requested time {t} is before trajectory start {self.t_min}. "
                         f"Clamping to {self.t_min}.")
            return self.t_min
        elif t > self.t_max:
            warnings.warn(f"Requested time {t} is after trajectory end {self.t_max}. "
                         f"Clamping to {self.t_max}.")
            return self.t_max
        return t
    
    def _interpolate_transform(self, t: float) -> fcl.Transform:
        """
        Interpolate transform at a given time.
        
        Uses linear interpolation for position and SLERP-like interpolation
        for rotation (simplified linear interpolation of rotation matrices).
        
        Args:
            t: Time at which to interpolate
            
        Returns:
            Interpolated fcl.Transform
        """
        # Clamp time to valid range
        t = self._clamp_time(t)
        
        # Find the two keyframes to interpolate between
        if t <= self.trajectory[0][1]:
            return self.trajectory[0][0]
        if t >= self.trajectory[-1][1]:
            return self.trajectory[-1][0]
        
        # Binary search for the correct interval
        left_idx = 0
        right_idx = len(self.trajectory) - 1
        
        for i in range(len(self.trajectory) - 1):
            if self.trajectory[i][1] <= t <= self.trajectory[i + 1][1]:
                left_idx = i
                right_idx = i + 1
                break
        
        # Get transforms and times
        transform_left, time_left = self.trajectory[left_idx]
        transform_right, time_right = self.trajectory[right_idx]
        
        # Compute interpolation parameter
        if time_right == time_left:
            alpha = 0.0
        else:
            alpha = (t - time_left) / (time_right - time_left)
        
        # Interpolate position
        pos_left = transform_left.getTranslation()
        pos_right = transform_right.getTranslation()
        pos_interp = pos_left + alpha * (pos_right - pos_left)
        
        # Interpolate rotation (simple linear interpolation of rotation matrices)
        rot_left = transform_left.getRotation()
        rot_right = transform_right.getRotation()
        rot_interp = rot_left + alpha * (rot_right - rot_left)
        
        # Orthonormalize the interpolated rotation matrix
        # Use modified Gram-Schmidt to ensure it remains a valid rotation
        rot_interp = self._orthonormalize(rot_interp)
        
        return fcl.Transform(rot_interp, pos_interp)
    
    @staticmethod
    def _orthonormalize(matrix: np.ndarray) -> np.ndarray:
        """
        Orthonormalize a 3x3 matrix using modified Gram-Schmidt.
        
        Args:
            matrix: 3x3 numpy array
            
        Returns:
            Orthonormalized 3x3 rotation matrix
        """
        # Use SVD for robust orthonormalization
        U, _, Vt = np.linalg.svd(matrix)
        return U @ Vt
    
    def get_transform(self, t: float = 0.0) -> fcl.Transform:
        """
        Get the interpolated transform at a specific time.
        
        Args:
            t: Time at which to query the obstacle (default: 0.0)
            
        Returns:
            fcl.Transform at time t (interpolated if between keyframes)
        """
        return self._interpolate_transform(t)
    
    def get_collision_object(self, t: float = 0.0) -> fcl.CollisionObject:
        """
        Get the collision object at a specific time.
        
        Args:
            t: Time at which to query the obstacle (default: 0.0)
            
        Returns:
            fcl.CollisionObject with transform interpolated to time t
        """
        transform = self.get_transform(t)
        return fcl.CollisionObject(self.geometry, transform)
