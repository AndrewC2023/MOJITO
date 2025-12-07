import numpy as np
from typing import Tuple, Optional, Union, Type
import sys
from pathlib import Path
import fcl

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Add gncpy dependency to path
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir / "dependencies" / "gncpy" / "src"))

import gncpy.math as gmath
from Utils.GeometryUtils import DCM3D


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
            Dictionary mapping 'position' (required) and optionally 'dcm' to state indices.
            'position': [x_idx, y_idx, z_idx] - indices for position in state vector
            'dcm': [i0, i1, ..., i8] - indices for 9-element DCM (row-major) in state vector
            Example: {'position': [0, 1, 2], 'dcm': [9, 10, 11, 12, 13, 14, 15, 16, 17]}
            If None, assumes position at [0, 1, 2] with no rotation (identity DCM).
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
            self._dcm_idx = None
            self._has_dcm = False
        else:
            self._pos_idx = np.array(state_indices.get('position', [0, 1, 2]), dtype=np.int32)
            dcm = state_indices.get('dcm')
            if dcm is not None:
                if len(dcm) != 9:
                    raise ValueError(f"DCM indices must have exactly 9 elements (row-major 3x3), got {len(dcm)}")
                self._dcm_idx = np.array(dcm, dtype=np.int32)
                self._has_dcm = True
            else:
                self._dcm_idx = None
                self._has_dcm = False
        
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
        For quaternion-based dynamics (SimpleMultirotorQuat), extracts quaternion,
        converts to Euler angles, and builds DCM for proper rotation.
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
        
        # Extract rotation - try multiple methods in order of preference
        rotation = np.eye(3)

        # TODO: this is highly dependent on the state vector dictionalry from the class
        # managing the dynamics. mazybe we need to fix this, proabaly rotation idx like position above
        
        # Check if model has quaternion state (SimpleMultirotorQuat)
        if hasattr(self.model, 'state_map'):
            state_map = self.model.state_map
            if hasattr(state_map, 'quat'):
                # Extract quaternion [qw, qx, qy, qz]
                quat = state_flat[state_map.quat]
                
                # Convert quaternion to Euler angles (roll, pitch, yaw)
                roll, pitch, yaw = gmath.quat_to_euler(quat)
                
                # Build DCM using ZYX convention (rotation = Rz @ Ry @ Rx)
                # This matches gncpy's quaternion convention and NED frame
                Rx = DCM3D(roll, "x")
                Ry = DCM3D(pitch, "y")
                Rz = DCM3D(yaw, "z")
                rotation = Rz @ Ry @ Rx
        
        # Use DCM indices if provided (for other dynamics models)
        elif self._has_dcm:
            # Extract 9 DCM elements and reshape to 3x3 (row-major)
            dcm_data = state_flat[self._dcm_idx]
            rotation = dcm_data.reshape(3, 3)
        
        # Update collision object transform
        transform = fcl.Transform(rotation, position)
        self.collision_object.setTransform(transform)
    

    
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
        if u is None:
            u = np.zeros(4)  # Default control for quadrotor
        
        # Try different propagate_state signatures
        # SimpleMultirotorQuat uses: propagate_state(timestep, state, u=motor_cmds)
        # SimpleMultirotor uses: propagate_state(motor_cmds, dt)
        # Generic DynamicsBase uses: propagate_state(dt, state, u=u)
        try:
            # Try SimpleMultirotorQuat signature first (timestep, state, u=u, **kwargs)
            self.state = self.model.propagate_state(dt, self.state, u=u, **kwargs)
        except TypeError:
            try:
                # Try SimpleMultirotor signature (motor_cmds, dt)
                self.state = self.model.propagate_state(u, dt)
            except TypeError:
                # Last resort: try (dt, state, u) without kwargs
                self.state = self.model.propagate_state(dt, self.state, u)
        
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
