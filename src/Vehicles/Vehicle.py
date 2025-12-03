import numpy as np
from typing import Tuple, Optional, Union, Type
import sys
from pathlib import Path
import fcl

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))


class Vehicle:
    """Vehicle class integrating gncpy dynamics with FCL collision geometry.
    
    Lightweight wrapper that holds a gncpy dynamics model and provides an FCL
    collision object for efficient collision checking with ConfigurationSpace3D.
    
    Attributes
    ----------
    model : DynamicsBase
        The gncpy dynamics model (e.g., SimpleMultirotor, DoubleIntegrator).
        Access this directly for dynamics operations.
    collision_object : fcl.CollisionObject
        The FCL collision object for this vehicle
    state : np.ndarray
        Current state vector of the vehicle
    """
    
    def __init__(
        self, 
        dynamics_class: Union[Type, object],
        geometry: fcl.CollisionGeometry,
        initial_state: Optional[np.ndarray] = None,
        state_indices: Optional[dict] = None,
        **dynamics_kwargs
    ):
        """Initialize the Vehicle.
        
        Parameters
        ----------
        dynamics_class : Type or instance
            Either a gncpy dynamics class (e.g., SimpleMultirotor) with **dynamics_kwargs,
            or an already instantiated dynamics object. If a class, it will be instantiated
            with dynamics_kwargs (e.g., params_file='path/to/config.yaml').
        geometry : fcl.CollisionGeometry
            FCL geometry for the vehicle (e.g., fcl.Box, fcl.Sphere, fcl.Cylinder).
        initial_state : np.ndarray, optional
            Initial state vector. If None, state must be set later.
        state_indices : dict, optional
            Dictionary mapping 'position' and optionally 'rotation' to state indices.
            Example: {'position': [0, 1, 2], 'rotation': [6, 7, 8]}
            If None, assumes position at [0, 1, 2].
        **dynamics_kwargs : dict
            Arguments passed to dynamics class constructor (e.g., params_file, env, etc.)
        """
        # Initialize or assign dynamics model
        if isinstance(dynamics_class, type):
            # It's a class, instantiate it with provided kwargs
            self.model = dynamics_class(**dynamics_kwargs)
        else:
            # It's already an instance
            self.model = dynamics_class
        
        # Store geometry
        self.geometry = geometry
        
        # Cache state indices for fast access
        if state_indices is None:
            self._pos_idx = np.array([0, 1, 2], dtype=np.int32)
            self._rot_idx = None
            self._has_rotation = False
        else:
            self._pos_idx = np.array(state_indices.get('position', [0, 1, 2]), dtype=np.int32)
            rot = state_indices.get('rotation')
            if rot is not None:
                self._rot_idx = np.array(rot) if isinstance(rot, list) else rot
                self._has_rotation = True
            else:
                self._rot_idx = None
                self._has_rotation = False
        
        # Initialize state
        self.state = np.asarray(initial_state).flatten() if initial_state is not None else None
        
        # Create FCL collision object at origin (will be updated)
        self.collision_object = fcl.CollisionObject(
            self.geometry,
            fcl.Transform(np.eye(3), [0.0, 0.0, 0.0])
        )
        
        # Update collision object if we have initial state
        if self.state is not None:
            self._update_collision_object_from_state()
    
    def _update_collision_object_from_state(self):
        """Internal optimized method to update collision object transform from state.
        
        Uses cached indices for fast extraction without creating temporary objects.
        Handles 2D states by setting z=0.
        """
        # Ensure state is flattened for indexing
        state_flat = self.state.flatten()
        
        # Extract position using cached indices
        # Handle 2D case (only x, y available)
        if len(self._pos_idx) == 2:
            position = np.array([
                state_flat[self._pos_idx[0]],
                state_flat[self._pos_idx[1]],
                0.0  # z = 0 for 2D states
            ], dtype=np.float64)
        else:
            position = np.array([
                state_flat[self._pos_idx[0]],
                state_flat[self._pos_idx[1]],
                state_flat[self._pos_idx[2]]
            ], dtype=np.float64)
        
        # Extract rotation if available
        rotation = np.eye(3)
        if self._has_rotation:
            rot_data = state_flat[self._rot_idx]
            rot_len = len(rot_data)
            
            if rot_len == 9:
                # DCM - reshape to 3x3
                rotation = rot_data.reshape(3, 3)
            elif rot_len == 3:
                # Euler angles - convert to rotation matrix
                rotation = self._euler_to_dcm(rot_data[0], rot_data[1], rot_data[2])
        
        # Update collision object transform
        transform = fcl.Transform(rotation, position)
        self.collision_object.setTransform(transform)
    
    @staticmethod
    def _euler_to_dcm(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Euler to DCM conversion (ZYX convention).
        
        Parameters
        ----------
        roll : float
            Roll angle in radians
        pitch : float
            Pitch angle in radians
        yaw : float
            Yaw angle in radians
            
        Returns
        -------
        np.ndarray
            3 by 3 rotation matrix
        """
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        
        # Pre-compute common terms
        cy_sp = cy * sp
        sy_sp = sy * sp
        
        # Build DCM directly (ZYX convention)
        return np.array([
            [cy * cp, cy_sp * sr - sy * cr, cy_sp * cr + sy * sr],
            [sy * cp, sy_sp * sr + cy * cr, sy_sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr]
        ], dtype=np.float64)
    
    def propagate(
        self, 
        dt: float, 
        u: Optional[np.ndarray] = None,
        **kwargs
    ) -> np.ndarray:
        """Propagate vehicle state forward by dt and update collision object.
        
        Uses self.model.propagate_state() internally.
        
        Parameters
        ----------
        dt : float
            Time step for propagation
        u : np.ndarray, optional
            Control input vector
        **kwargs : dict
            Additional arguments for model.propagate_state()
            
        Returns
        -------
        np.ndarray
            The propagated state
        """
        # Propagate state using the dynamics model
        self.state = self.model.propagate_state(dt, self.state, u=u, **kwargs)
        
        # Update collision object transform
        self._update_collision_object_from_state()
        
        return self.state
    
    def set_state(self, state: np.ndarray):
        """Set vehicle state and update collision object.
        
        Parameters
        ----------
        state : np.ndarray
            New state vector
        """
        self.state = np.asarray(state).flatten()
        self._update_collision_object_from_state()
    
    def get_position(self) -> np.ndarray:
        """Get current position as numpy array.
        
        Returns
        -------
        np.ndarray
            Position [x, y, z] or [x, y] for 2D states
        """
        state_flat = self.state.flatten()
        if len(self._pos_idx) == 2:
            return state_flat[self._pos_idx]
        else:
            return state_flat[self._pos_idx]
    
    def get_transform(self) -> fcl.Transform:
        """Get current FCL transform of the vehicle.
        
        Returns
        -------
        fcl.Transform
            Current transform of the collision object
        """
        return self.collision_object.getTransform()
    
    def __repr__(self) -> str:
        """String representation."""
        state_dim = len(self.state) if self.state is not None else "unset"
        geom_type = type(self.geometry).__name__
        return f"Vehicle(model={type(self.model).__name__}, geometry={geom_type}, state_dim={state_dim})"
