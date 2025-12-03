import numpy as np
from scipy.interpolate import PchipInterpolator
from gncpy.dynamics.basic import DynamicsBase
from Vehicles.Vehicle import Vehicle
from Optimizers.OptimizerBase import OptimizerBase, CostFunction

"""in all these cases we are assuming normalized u values between 0 and 1, the vehicle model will need to scale them appropriately """


class inputFunction:
    """this is a class that defines the ionterface for input functions used in NACMPC
        to reduce dimentions we can have the input function define how to map from keyframes to full control inputs"""
    def __init__(self, numKeyframes: int, totalSteps: int, **kwargs):
        self.numKeyframes = numKeyframes
        self.totalSteps = totalSteps


    def updateKeyFrameValues(self, keyframeValues: np.ndarray) -> np.ndarray:
        """Update the keyframe values to full control inputs.
        
        Parameters
        ----------
        keyframeValues : np.ndarray
            Control inputs at keyframes.
        
        Returns
        -------
        np.ndarray
            Full control inputs for all time steps.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    def updateTimeStep(self, dt: float):
        """Update the total number of time steps.
        
        Parameters
        ----------
        totalSteps : int
            New total number of time steps.
        """
        self.dt = dt
    def updateStartAndEndTimes(self, startTime: float, endTime: float):
        """Update the start and end times for the control inputs.
        
        Parameters
        ----------
        startTime : float
            Start time.
        endTime : float
            End time.
        """
        self.startTime = startTime
        self.endTime = endTime

    def calculateInput(self, time: float) -> np.ndarray:
        """Calculate control input at a specific time.
            This is where the control function should be implemented.
        
        Parameters
        ----------
        time : float
            Time at which to calculate control input.
        
        Returns
        -------
        np.ndarray
            Control input at the specified time.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

class PiecewiseConstantInput(inputFunction):
    """Zero-order hold: control stays constant from one keyframe to the next.
    
    This is the simplest interpolation scheme. The control at time t equals
    the value of the most recent keyframe. No interpolation between keyframes.
    
    Example with 3 keyframes over [0, 10]:
        keyframe_times = [0, 5, 10]
        u(t) = keyframes[0] for t in [0, 5)
        u(t) = keyframes[1] for t in [5, 10)
        u(t) = keyframes[2] for t = 10
    
    Parameters
    ----------
    u_min : float or np.ndarray, optional
        Minimum saturation limit(s). Scalar applies to all dimensions.
        Default: 0.0 (normalized controls)
    u_max : float or np.ndarray, optional
        Maximum saturation limit(s). Scalar applies to all dimensions.
        Default: 1.0 (normalized controls)
    """
    
    def __init__(self, numKeyframes: int, totalSteps: int, control_dim: int = 4, 
                 u_min: float = 0.0, u_max: float = 1.0, **kwargs):
        super().__init__(numKeyframes, totalSteps, **kwargs)
        self.control_dim = control_dim
        self.keyframeValues = None
        self.keyframe_times = None
        self.dt = 0.1
        self.startTime = 0.0
        self.endTime = 1.0
        
        # Saturation limits
        self.u_min = np.atleast_1d(u_min)
        self.u_max = np.atleast_1d(u_max)
        if self.u_min.size == 1:
            self.u_min = np.full(control_dim, self.u_min[0])
        if self.u_max.size == 1:
            self.u_max = np.full(control_dim, self.u_max[0])
        
        self._update_keyframe_times()
    
    def _update_keyframe_times(self):
        """Compute the times at which keyframes occur."""
        self.keyframe_times = np.linspace(self.startTime, self.endTime, self.numKeyframes)
    
    def updateKeyFrameValues(self, keyframeValues: np.ndarray) -> None:
        """Update the keyframe control values.
        
        Parameters
        ----------
        keyframeValues : np.ndarray
            Flattened array of shape (numKeyframes * control_dim,) or
            2D array of shape (numKeyframes, control_dim).
        """
        keyframeValues = np.asarray(keyframeValues)
        if keyframeValues.ndim == 1:
            # Reshape from flat to (numKeyframes, control_dim)
            self.keyframeValues = keyframeValues.reshape(self.numKeyframes, self.control_dim)
        else:
            self.keyframeValues = keyframeValues
    
    def updateStartAndEndTimes(self, startTime: float, endTime: float):
        """Update the start and end times and recompute keyframe times."""
        self.startTime = startTime
        self.endTime = endTime
        self._update_keyframe_times()
    
    def calculateInput(self, time: float) -> np.ndarray:
        """Calculate control input at a specific time using zero-order hold.
        
        Parameters
        ----------
        time : float
            Time at which to calculate control input.
        
        Returns
        -------
        np.ndarray
            Control input vector at the specified time.
        """
        if self.keyframeValues is None:
            raise ValueError("Keyframe values not set. Call updateKeyFrameValues first.")
        
        # Find the index of the most recent keyframe
        idx = np.searchsorted(self.keyframe_times, time, side='right') - 1
        idx = np.clip(idx, 0, self.numKeyframes - 1)
        
        # Apply saturation limits
        control = self.keyframeValues[idx].copy()
        return np.clip(control, self.u_min, self.u_max)


class LinearInterpolationInput(inputFunction):
    """Linear interpolation: control linearly interpolates between keyframes.
    
    The control at time t is a linear blend between adjacent keyframes.
    Provides first-order continuity (C0 continuous).
    
    Example with 3 keyframes at t=[0, 5, 10]:
        u(2.5) = 0.5 * keyframes[0] + 0.5 * keyframes[1]
        u(7.5) = 0.5 * keyframes[1] + 0.5 * keyframes[2]
    
    Parameters
    ----------
    u_min : float or np.ndarray, optional
        Minimum saturation limit(s). Scalar applies to all dimensions.
        Default: 0.0 (normalized controls)
    u_max : float or np.ndarray, optional
        Maximum saturation limit(s). Scalar applies to all dimensions.
        Default: 1.0 (normalized controls)
    """
    
    def __init__(self, numKeyframes: int, totalSteps: int, control_dim: int = 4,
                 u_min: float = 0.0, u_max: float = 1.0, **kwargs):
        super().__init__(numKeyframes, totalSteps, **kwargs)
        self.control_dim = control_dim
        self.keyframeValues = None
        self.keyframe_times = None
        self.dt = 0.1
        self.startTime = 0.0
        self.endTime = 1.0
        
        # Saturation limits
        self.u_min = np.atleast_1d(u_min)
        self.u_max = np.atleast_1d(u_max)
        if self.u_min.size == 1:
            self.u_min = np.full(control_dim, self.u_min[0])
        if self.u_max.size == 1:
            self.u_max = np.full(control_dim, self.u_max[0])
        
        self._update_keyframe_times()
    
    def _update_keyframe_times(self):
        """Compute the times at which keyframes occur."""
        self.keyframe_times = np.linspace(self.startTime, self.endTime, self.numKeyframes)
    
    def updateKeyFrameValues(self, keyframeValues: np.ndarray) -> None:
        """Update the keyframe control values.
        
        Parameters
        ----------
        keyframeValues : np.ndarray
            Flattened array of shape (numKeyframes * control_dim,) or
            2D array of shape (numKeyframes, control_dim).
        """
        keyframeValues = np.asarray(keyframeValues)
        if keyframeValues.ndim == 1:
            self.keyframeValues = keyframeValues.reshape(self.numKeyframes, self.control_dim)
        else:
            self.keyframeValues = keyframeValues
    
    def updateStartAndEndTimes(self, startTime: float, endTime: float):
        """Update the start and end times and recompute keyframe times."""
        self.startTime = startTime
        self.endTime = endTime
        self._update_keyframe_times()
    
    def calculateInput(self, time: float) -> np.ndarray:
        """Calculate control input at a specific time using linear interpolation.
        
        Parameters
        ----------
        time : float
            Time at which to calculate control input.
        
        Returns
        -------
        np.ndarray
            Control input vector at the specified time.
        """
        if self.keyframeValues is None:
            raise ValueError("Keyframe values not set. Call updateKeyFrameValues first.")
        
        # Clamp time to valid range
        time = np.clip(time, self.startTime, self.endTime)
        
        # Find the two keyframes to interpolate between
        idx = np.searchsorted(self.keyframe_times, time, side='right') - 1
        idx = np.clip(idx, 0, self.numKeyframes - 2)  # Ensure we have idx and idx+1
        
        # Get times and values for interpolation
        t0 = self.keyframe_times[idx]
        t1 = self.keyframe_times[idx + 1]
        u0 = self.keyframeValues[idx]
        u1 = self.keyframeValues[idx + 1]
        
        # Linear interpolation parameter
        alpha = (time - t0) / (t1 - t0) if t1 > t0 else 0.0
        alpha = np.clip(alpha, 0.0, 1.0)
        
        # Interpolate and apply saturation limits
        control = (1.0 - alpha) * u0 + alpha * u1
        return np.clip(control, self.u_min, self.u_max)


class SplineInterpolationInput(inputFunction):
    """PCHIP (shape-preserving cubic Hermite) interpolation for smooth control.
    
    Uses Piecewise Cubic Hermite Interpolating Polynomial (PCHIP) which provides:
    - C1 continuity (continuous first derivatives)
    - Shape-preserving: stays within convex hull of keyframes
    - NO overshoot between keyframes (monotonic where data is monotonic)
    - Much better behaved than natural cubic splines for control inputs
    
    PCHIP is fast (similar performance to cubic splines) and prevents the
    undesirable "wiggling" and overshoot that standard cubic splines exhibit.
    
    Parameters
    ----------
    u_min : float or np.ndarray, optional
        Minimum saturation limit(s). Scalar applies to all dimensions.
        Default: 0.0 (normalized controls)
    u_max : float or np.ndarray, optional
        Maximum saturation limit(s). Scalar applies to all dimensions.
        Default: 1.0 (normalized controls)
    """
    
    def __init__(self, numKeyframes: int, totalSteps: int, control_dim: int = 4,
                 u_min: float = 0.0, u_max: float = 1.0, **kwargs):
        super().__init__(numKeyframes, totalSteps, **kwargs)
        self.control_dim = control_dim
        self.keyframeValues = None
        self.keyframe_times = None
        self.pchip_interpolators = None  # List of PchipInterpolator objects
        self.dt = 0.1
        self.startTime = 0.0
        self.endTime = 1.0
        
        # Saturation limits
        self.u_min = np.atleast_1d(u_min)
        self.u_max = np.atleast_1d(u_max)
        if self.u_min.size == 1:
            self.u_min = np.full(control_dim, self.u_min[0])
        if self.u_max.size == 1:
            self.u_max = np.full(control_dim, self.u_max[0])
        
        self._update_keyframe_times()
    
    def _update_keyframe_times(self):
        """Compute the times at which keyframes occur."""
        self.keyframe_times = np.linspace(self.startTime, self.endTime, self.numKeyframes)
    
    def updateKeyFrameValues(self, keyframeValues: np.ndarray) -> None:
        """Update the keyframe control values and construct PCHIP interpolators.
        
        Parameters
        ----------
        keyframeValues : np.ndarray
            Flattened array of shape (numKeyframes * control_dim,) or
            2D array of shape (numKeyframes, control_dim).
        """
        keyframeValues = np.asarray(keyframeValues)
        if keyframeValues.ndim == 1:
            self.keyframeValues = keyframeValues.reshape(self.numKeyframes, self.control_dim)
        else:
            self.keyframeValues = keyframeValues
        
        # Construct PCHIP interpolators for each control dimension
        self.pchip_interpolators = []
        for i in range(self.control_dim):
            # Extract values for this control dimension across all keyframes
            values = self.keyframeValues[:, i]
            # Create PCHIP interpolator (shape-preserving, no overshoot)
            pchip = PchipInterpolator(self.keyframe_times, values)
            self.pchip_interpolators.append(pchip)
    
    def updateStartAndEndTimes(self, startTime: float, endTime: float):
        """Update the start and end times and recompute keyframe times.
        
        Note: This requires reconstructing the interpolators if keyframe values exist.
        """
        self.startTime = startTime
        self.endTime = endTime
        self._update_keyframe_times()
        
        # Reconstruct interpolators with new time basis
        if self.keyframeValues is not None:
            self.updateKeyFrameValues(self.keyframeValues)
    
    def calculateInput(self, time: float) -> np.ndarray:
        """Calculate control input at a specific time using PCHIP interpolation.
        
        Parameters
        ----------
        time : float
            Time at which to calculate control input.
        
        Returns
        -------
        np.ndarray
            Control input vector at the specified time.
        """
        if self.pchip_interpolators is None:
            raise ValueError("Keyframe values not set. Call updateKeyFrameValues first.")
        
        # Clamp time to valid range
        time = np.clip(time, self.startTime, self.endTime)
        
        # Evaluate each PCHIP interpolator at the given time
        control = np.array([pchip(time) for pchip in self.pchip_interpolators])
        
        # Apply saturation limits
        return np.clip(control, self.u_min, self.u_max)


class NACMPC:
    """Nonlinear Arbitrary Cost Model Predictive Controller (NAC-MPC).

    This class owns the dynamics rollout and cost evaluation. Optimizers see
    it as a black box that maps a decision vector (keyframe values and
    optionally horizon) to a scalar cost via ``evaluate_decision_vector``.
    """

    def __init__(
        self,
        vehicle: Vehicle,
        costFunction: CostFunction,
        optimizer: OptimizerBase,
        inputFunction: inputFunction,
        control_dim: int = 4,  # Number of control inputs (e.g., 4 rotors)
        debug: bool = False,   # Enable debug prints
        **kwargs,
    ):
        self.vehicle = vehicle
        self.costFunction = costFunction
        self.optimizer = optimizer
        self.inputFunction = inputFunction
        self.control_dim = control_dim
        self.debug = debug
        self.eval_count = 0  # Track number of evaluations

        # default parameters
        self.physicsSteps = 1000  # number of integration steps per rollout
        self.numControlKeyframes = 100  # dimensionality of control keyframes
        self.dynamicHorizon = True  # optimizer can change horizon length
        self.maxHorizon = 15.0  # seconds
        self.minHorizon = 1.0  # seconds

        # extra context passed into cost function (config space, goal, etc.)
        self.cost_context = {}

        # keyword arguments to override defaults
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        if self.debug:
            print(f"[NACMPC.__init__] control_dim={self.control_dim}, numKeyframes={self.numControlKeyframes}, physicsSteps={self.physicsSteps}")

    # ---- public API used by user code ----

    def plan(self, x0: np.ndarray, initial_decision: np.ndarray, **cost_context):
        """Run MPC planning and return the optimized decision vector.

        Parameters
        ----------
        x0 : np.ndarray
            Initial state of the system.
        initial_decision : np.ndarray
            Initial guess for decision vector (keyframes [+ horizon]).
        **cost_context : dict
            Extra info passed to the cost function (goal, config_space, etc.).

        Returns
        -------
        np.ndarray
            Optimized decision vector (same structure as ``initial_decision``).
        """
        self._x0 = np.asarray(x0).copy()
        self.cost_context = cost_context

        return self.optimizer.optimize(initial_decision, controller=self)

    # ---- interface used by optimizers ----

    def evaluate_decision_vector(self, decision_vector: np.ndarray) -> float:
        """Roll out dynamics for a candidate decision vector and return cost.

        The decision vector layout is:

        - If ``dynamicHorizon`` is True: ``[horizon, kf_0, kf_1, ..., kf_N]``
        - Otherwise: ``[kf_0, kf_1, ..., kf_N]``

        The exact interpretation of keyframes is delegated to ``inputFunction``.
        """
        self.eval_count += 1
        verbose = self.debug and (self.eval_count <= 2)  # Only print first 2 evaluations
        
        decision_vector = np.asarray(decision_vector, dtype=float)
        
        if verbose:
            print(f"\n[NACMPC.evaluate_decision_vector] #{self.eval_count}")
            print(f"  decision_vector.shape={decision_vector.shape}")
            print(f"  decision_vector range: [{decision_vector.min():.4f}, {decision_vector.max():.4f}]")
        elif self.debug and self.eval_count % 20 == 0:
            print(f"[NACMPC] Evaluation #{self.eval_count}...")

        # decode horizon and keyframes
        if self.dynamicHorizon:
            horizon = float(decision_vector[0])
            # clamp to allowed range
            horizon = max(self.minHorizon, min(self.maxHorizon, horizon))
            keyframes = decision_vector[1:]
            if verbose:
                print(f"  Dynamic horizon: {horizon:.3f}s (raw: {decision_vector[0]:.3f})")
                print(f"  keyframes.shape={keyframes.shape}, range=[{keyframes.min():.4f}, {keyframes.max():.4f}]")
        else:
            horizon = self.maxHorizon
            keyframes = decision_vector
            if verbose:
                print(f"  Fixed horizon: {horizon:.3f}s, keyframes.shape={keyframes.shape}")

        # set up time discretization
        total_steps = self.physicsSteps
        dt = horizon / total_steps
        
        if verbose:
            print(f"  Rollout: {total_steps} steps, dt={dt:.6f}s")

        # configure input function
        self.inputFunction.updateTimeStep(dt)
        self.inputFunction.updateStartAndEndTimes(0.0, horizon)
        self.inputFunction.updateKeyFrameValues(keyframes)
        
        if verbose:
            print(f"  Input function configured: {type(self.inputFunction).__name__}")

        # roll out dynamics from x0
        # make a copy of the vehicle state so optimization is side-effect free
        original_state = None if self.vehicle.state is None else self.vehicle.state.copy()
        if original_state is None:
            self.vehicle.set_state(self._x0)
        else:
            self.vehicle.set_state(self._x0)
        
        if verbose:
            print(f"  Initial state set: {self._x0[:3]} (showing pos only)")

        total_cost = 0.0
        t = 0.0
        for step in range(total_steps):
            u = self.inputFunction.calculateInput(t)
            
            if verbose and (step == 0 or step == total_steps - 1):
                print(f"  Step {step}/{total_steps}: t={t:.4f}s, u={u}")
            
            state = self.vehicle.propagate(dt, u=u)
            
            if verbose and (step == 0 or step == total_steps - 1):
                pos = state[:3] if len(state) >= 3 else state
                print(f"    -> pos={pos}")
            
            # Check if this is the terminal step
            is_terminal = (step == total_steps - 1)
            
            # Get collision query result from configuration space
            # Pass the vehicle's FCL collision object, not the Vehicle wrapper
            collision_result = self.cost_context.get('config_space').query_collision_detailed(
                self.vehicle.collision_object, t
            )
            
            if verbose and (step == 0 or step == total_steps - 1):
                print(f"    -> collision: {collision_result.has_collision}, dist={collision_result.min_obstacle_distance:.4f}")
            
            instant_cost = self.costFunction.evaluate(
                state=state,
                control=u,
                dt=dt,
                collision_result=collision_result,
                is_terminal=is_terminal,
                **self.cost_context,
            )
            
            if verbose and (step == 0 or step == total_steps - 1):
                print(f"    -> instant_cost={instant_cost:.4f}")
            
            total_cost += float(instant_cost)
            t += dt
        
        if verbose:
            print(f"  ROLLOUT COMPLETE: total_cost={total_cost:.4f}\n")

        # restore original vehicle state
        if original_state is not None:
            self.vehicle.set_state(original_state)

        return total_cost

        



        