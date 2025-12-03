# This is the Test script for the test case of trying to have
# a quad coptor fly through a narrow vertical slit in a wall
# the solver is the NMPC (non-linear model predictive controller) with
# the optimizer for hig dimensional problems as we also include difficult
# to evaluate cost functions (We want this to handle arbitrary cost function
# setups), in that vain we need optimizers that see the cost function as a
# black box and do not require gradients or hessians especially considering
# we want to consider discontinuous cost functions.

import matplotlib.pyplot as plt
import numpy as np
import fcl

import sys
from pathlib import Path

# Add project source and gncpy dependency to path
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir / "src"))
sys.path.append(str(root_dir / "dependencies" / "gncpy" / "src"))

from gncpy.dynamics.aircraft.simple_multirotor_quat import SimpleMultirotorQuat, v_smap_quat
from gncpy.dynamics.aircraft.simple_multirotor import Effector
import gncpy.math as gmath
import fcl
from Utils.GeometryUtils import DCM3D  # type: ignore
from ConfigurationSpace.ConfigSpace3D import ConfigurationSpace3D # type: ignore
from ConfigurationSpace.Obstacles import StaticObstacle # type: ignore
from Vehicles.Vehicle import Vehicle # type: ignore
from Controls.NACMPC import NACMPC # type: ignore
from Optimizers.OptimizerBase import OptimizerBase, CostFunction # type: ignore
from Optimizers.CrossEntropyMethod import CrossEntropyMethod # type: ignore

# configuration setup (Note we are in NED frame - this makes dynamics frame conversion easier for me)
# NED: X=North, Y=East, Z=Down (negative is up)
# zMin should be more negative (deeper/lower), zMax less negative (higher/surface)
dim = [0, 10, -2.5, 2.5, -5, 0]  # xMin, xMax, yMin, yMax, zMin, zMax

GRAVITY = np.array([0,0,9.81])  # acceleration vector in NED frame

# construct the dynamics class
config_file = Path(__file__).parent / "SmallQuadrotor.yaml"
QuadDynamics = SimpleMultirotorQuat(str(config_file), effector=None)
print("quad constructed")
# Initialize vehicle with proper NED setup
INIT_POS = np.array([1.5, -1.25, -1.5])  # NED (m)
INIT_VEL = np.array([0.0, 0.0, 0.0])  # body frame (m/s)
INIT_EULER = np.array([0.0, 0.0, 0.0])  # roll, pitch, yaw (deg)
INIT_RATES = np.array([0.0, 0.0, 0.0])  # body rates (rad/s)
REF_LAT, REF_LON, TERRAIN_ALT = 34.0, -86.0, 0.0
ned_mag = np.array([20.0, 5.0, 45.0])

QuadDynamics.set_initial_conditions(
    INIT_POS, INIT_VEL, INIT_EULER, INIT_RATES, REF_LAT, REF_LON, TERRAIN_ALT, ned_mag
)
QuadDynamics.vehicle.takenoff = True

print(f"\n=== Vehicle State Initialization ===")
print(f"State shape: {QuadDynamics.vehicle.state.shape}")
print(f"NED position: {QuadDynamics.vehicle.state[v_smap_quat.ned_pos]}")
print(f"Body velocity: {QuadDynamics.vehicle.state[v_smap_quat.body_vel]}")
print(f"Quaternion: {QuadDynamics.vehicle.state[v_smap_quat.quat]}")
print(f"Body rates: {QuadDynamics.vehicle.state[v_smap_quat.body_rot_rate]}")
print(f"State vector length: {len(QuadDynamics.vehicle.state)}")

# Vehicle geometry (0.3m x 0.3m x 0.1m)
vehicle_geometry = fcl.Box(0.3, 0.3, 0.1)

# Initial state (13-DOF with quaternions: pos(3), vel(3), quat(4), rates(3))
x0 = QuadDynamics.vehicle.state.copy()

# Goal position (on the other side of the slit)
goal_position = np.array([8.5, 1.25, -3.5])  # [x, y, z] in NED

# Goal state (13-DOF)
x_goal = x0.copy()
x_goal[v_smap_quat.ned_pos] = goal_position
x_goal[v_smap_quat.body_vel] = np.zeros(3)  # zero velocity at goal
x_goal[v_smap_quat.body_rot_rate] = np.zeros(3)  # zero rates at goal

print(f"\nGoal position: {goal_position}")
print(f"Goal state shape: {x_goal.shape}")

# Construct vehicle with correct state indices for SimpleMultirotorQuat
# Vehicle class expects position only; we'll handle rotation externally
vehicle = Vehicle(
    dynamics_class=QuadDynamics,
    geometry=vehicle_geometry,
    initial_state=x0,
    state_indices={
        'position': [4, 5, 6]  # v_smap_quat.ned_pos
    }
)

"""Set up configuration space with a vertical wall containing a narrow slit.

The domain is given by ``dim`` (xMin, xMax, yMin, yMax, zMin, zMax).
We build three static obstacles that form a solid wall in X, with a
vertical slit in Y/Z sized by ``slit_width`` and ``slit_height``.
"""

# make the configuration space
config_space = ConfigurationSpace3D(dim)

slit_width = 0.25   # opening width in Y (meters)
slit_height = 1   # opening height in Z (meters)
wall_thickness = 1  # wall thickness in X (meters)

x_center = 0.5 * (dim[0] + dim[1])
y_mid = 0.5 * (dim[2] + dim[3])
z_min = dim[4]
z_max = dim[5]
z_mid = 0.5 * (z_min + z_max)

total_y = dim[3] - dim[2]
upper_y_height = 0.5 * total_y - 0.5 * slit_width

total_z = abs(z_max - z_min)
lower_z_height = 0.5 * total_z - 0.5 * slit_height
upper_z_height = 0.5 * total_z - 0.5 * slit_height

# OBSTACLES COMMENTED OUT FOR DEBUGGING - just fly from start to goal without obstacles
# Left side of slit (negative Y side) - full height wall
# left_geom = fcl.Box(wall_thickness, upper_y_height, total_z)
# left_tf = fcl.Transform(np.eye(3), [x_center, dim[2] + 0.5 * upper_y_height, z_mid])
# config_space.add_obstacle(StaticObstacle(left_geom, left_tf))

# Right side of slit (positive Y side) - full height wall
# right_geom = fcl.Box(wall_thickness, upper_y_height, total_z)
# right_tf = fcl.Transform(np.eye(3), [x_center, dim[3] - 0.5 * upper_y_height, z_mid])
# config_space.add_obstacle(StaticObstacle(right_geom, right_tf))

# Bottom of slit (across slit width at lower Z region)
# bottom_geom = fcl.Box(wall_thickness, slit_width, lower_z_height)
# bottom_tf = fcl.Transform(np.eye(3), [x_center, y_mid, z_mid - 0.5 * slit_height - 0.5 * lower_z_height])
# config_space.add_obstacle(StaticObstacle(bottom_geom, bottom_tf))

# Top of slit (across slit width at upper Z region)
# top_geom = fcl.Box(wall_thickness, slit_width, upper_z_height)
# top_tf = fcl.Transform(np.eye(3), [x_center, y_mid, z_mid + 0.5 * slit_height + 0.5 * upper_z_height])
# config_space.add_obstacle(StaticObstacle(top_geom, top_tf))

# Define cost function class for the optimizer
class QuadSlitCostFunction(CostFunction):
    """Cost function for quadcopter flying through slit in wall.
    
    This cost function combines:
    1. LQR-style quadratic penalties on state and control deviations
    2. Discrete collision penalties (proportional to penetration depth)
    3. Boundary violation penalties
    4. Soft proximity constraints (keeps vehicle away from obstacles)
    5. Goal tracking penalties
    6. Terminal cost (large penalty if final state is far from goal)
    
    Parameters
    ----------
    goal_state : np.ndarray
        Target state vector (position, velocity, attitude, rates)
    goal_position : np.ndarray
        Target position [x, y, z] for proximity calculations
    Q : np.ndarray
        State penalty matrix (diagonal or full)
    R : np.ndarray
        Control input penalty matrix (diagonal or full)
    collision_weight : float
        Weight for collision penetration depth penalty
    boundary_weight : float
        Weight for boundary violation penalty
    proximity_weight : float
        Weight for obstacle proximity penalty
    proximity_threshold : float
        Distance threshold for proximity penalties (meters)
    goal_weight : float
        Weight for goal position tracking
    terminal_goal_weight : float
        Weight for terminal goal proximity (used when is_terminal=True)
    """

    def __init__(self, 
                 goal_state: np.ndarray,
                 goal_position: np.ndarray,
                 Q: np.ndarray,
                 R: np.ndarray,
                 collision_weight: float = 1000.0,
                 boundary_weight: float = 1000.0,
                 proximity_weight: float = 10.0,
                 proximity_threshold: float = 0.5,
                 goal_weight: float = 50.0,
                 terminal_goal_weight: float = 500.0,
                 debug: bool = False):
        super().__init__()
        self.goal_state = goal_state
        self.goal_position = goal_position
        self.Q = Q
        self.R = R
        self.collision_weight = collision_weight
        self.boundary_weight = boundary_weight
        self.proximity_weight = proximity_weight
        self.proximity_threshold = proximity_threshold
        self.goal_weight = goal_weight
        self.terminal_goal_weight = terminal_goal_weight
        self.debug = debug
        self.eval_count = 0

    def evaluate(self, state: np.ndarray, control: np.ndarray, dt: float, **kwargs) -> float:
        """Evaluate cost for a single state-control pair.
        
        Parameters
        ----------
        state : np.ndarray
            Vehicle state vector (48-DOF from SimpleMultirotorQuat)
        control : np.ndarray
            Control input vector
        dt : float
            Time step
        **kwargs : dict
            Additional arguments (collision_result, is_terminal, goal_state, etc.)
            
        Returns
        -------
        float
            Total cost for this state-control pair
        """
        self.eval_count += 1
        is_terminal = kwargs.get('is_terminal', False)
        collision_result = kwargs.get('collision_result', None)
        
        # VALIDATION: Check input dimensions
        if self.eval_count == 1:
            print(f"\n[COST FUNCTION DIMENSION VALIDATION]")
            print(f"  state type: {type(state)}, shape: {state.shape}, dtype: {state.dtype}")
            print(f"  control type: {type(control)}, shape: {control.shape}, dtype: {control.dtype}")
            print(f"  goal_state shape: {self.goal_state.shape}")
            print(f"  Q matrix shape: {self.Q.shape}")
            print(f"  R matrix shape: {self.R.shape}")
        
        # Extract 12-DOF state we care about: [pos(3), vel(3), euler(3), rates(3)]
        # Position: NED position
        pos = state[v_smap_quat.ned_pos].flatten()  # indices [4,5,6], FLATTEN to 1D
        # Velocity: body frame velocity
        vel = state[v_smap_quat.body_vel].flatten()  # indices [17,18,19], FLATTEN to 1D
        # Attitude: convert quaternion to Euler angles
        quat = state[v_smap_quat.quat].flatten()  # indices [13,14,15,16], FLATTEN to 1D
        roll, pitch, yaw = gmath.quat_to_euler(quat)
        euler = np.array([roll, pitch, yaw]).flatten()
        # Body rates
        rates = state[v_smap_quat.body_rot_rate].flatten()  # indices [23,24,25], FLATTEN to 1D
        
        if self.eval_count == 1:
            print(f"  Extracted components:")
            print(f"    pos shape: {pos.shape}, values: {pos}")
            print(f"    vel shape: {vel.shape}, values: {vel}")
            print(f"    quat shape: {quat.shape}, values: {quat}")
            print(f"    euler shape: {euler.shape}, values: {euler}")
            print(f"    rates shape: {rates.shape}, values: {rates}")
        
        # Build 12-DOF state vector for cost computation
        state_12dof = np.concatenate([pos, vel, euler, rates])
        
        if self.eval_count == 1:
            print(f"  state_12dof shape: {state_12dof.shape}, should be (12,)")
            if state_12dof.shape != (12,):
                raise ValueError(f"state_12dof has wrong shape: {state_12dof.shape}, expected (12,)")
        
        # Extract same 12-DOF from goal state
        goal_pos = self.goal_state[v_smap_quat.ned_pos].flatten()
        goal_vel = self.goal_state[v_smap_quat.body_vel].flatten()
        goal_quat = self.goal_state[v_smap_quat.quat].flatten()
        goal_roll, goal_pitch, goal_yaw = gmath.quat_to_euler(goal_quat)
        goal_euler = np.array([goal_roll, goal_pitch, goal_yaw]).flatten()
        goal_rates = self.goal_state[v_smap_quat.body_rot_rate].flatten()
        goal_12dof = np.concatenate([goal_pos, goal_vel, goal_euler, goal_rates])
        
        if self.eval_count == 1:
            print(f"  goal_12dof shape: {goal_12dof.shape}, should be (12,)")
            if goal_12dof.shape != (12,):
                raise ValueError(f"goal_12dof has wrong shape: {goal_12dof.shape}, expected (12,)")
        
        # State error cost (LQR-style quadratic penalty on 12-DOF)
        state_error = state_12dof - goal_12dof
        
        if self.eval_count == 1:
            print(f"  state_error shape: {state_error.shape}, should be (12,)")
            print(f"  Q @ state_error shape: {(self.Q @ state_error).shape}")
            print(f"  Computing state_cost = state_error.T @ Q @ state_error")
        
        state_cost = np.dot(state_error, self.Q @ state_error)
        
        if self.eval_count == 1:
            print(f"  state_cost type: {type(state_cost)}, value: {state_cost}")
            if not np.isscalar(state_cost) and hasattr(state_cost, 'shape'):
                print(f"  WARNING: state_cost is not scalar! shape: {state_cost.shape}")
        
        # Control effort cost
        control_flat = control.flatten()
        
        if self.eval_count == 1:
            print(f"  control original shape: {control.shape}")
            print(f"  control_flat shape: {control_flat.shape}, should be (4,)")
            print(f"  R @ control_flat shape: {(self.R @ control_flat).shape}")
        
        control_cost = np.dot(control_flat, self.R @ control_flat)
        
        if self.eval_count == 1:
            print(f"  control_cost type: {type(control_cost)}, value: {control_cost}")
            if not np.isscalar(control_cost) and hasattr(control_cost, 'shape'):
                print(f"  WARNING: control_cost is not scalar! shape: {control_cost.shape}")
        
        # Collision penalty (exponential in penetration depth)
        collision_cost = 0.0
        if collision_result and collision_result.has_collision:
            penetration = collision_result.total_penetration_depth
            collision_cost = self.collision_weight * (np.exp(penetration) - 1.0)
        
        # Boundary violation penalty (very high to absolutely avoid)
        boundary_cost = 0.0
        if collision_result and collision_result.is_out_of_bounds:
            boundary_cost = self.boundary_weight
        
        # Proximity penalty (log barrier to stay away from obstacles)
        proximity_cost = 0.0
        if collision_result and not collision_result.has_collision:
            dist = collision_result.min_obstacle_distance
            if dist < self.proximity_threshold:
                # Log barrier: cost grows as we approach threshold
                proximity_cost = -self.proximity_weight * np.log(dist / self.proximity_threshold)
        
        # Goal tracking cost (distance to goal position)
        goal_dist = np.linalg.norm(pos - self.goal_position)
        goal_cost = self.goal_weight * goal_dist
        
        # Terminal cost (large penalty if far from goal at end)
        terminal_cost = 0.0
        if is_terminal:
            terminal_cost = self.terminal_goal_weight * goal_dist
        
        # Total cost
        total = state_cost + control_cost + collision_cost + boundary_cost + proximity_cost + goal_cost + terminal_cost
        
        # Debug output for occasional evaluations
        if self.debug:
            # Print every 50 evaluations or if terminal
            if is_terminal or (self.eval_count % 100 == 0):
                obst_dist = collision_result.min_obstacle_distance if collision_result else float('inf')
                print(f"\n[Cost eval #{self.eval_count}]{' TERMINAL' if is_terminal else ''}", flush=True)
                print(f"  Pos: {pos.T[0]}, Goal dist: {goal_dist:.2f}m, Obst dist: {obst_dist:.2f}m", flush=True)
                print(f"  Costs: state={state_cost:.1f}, ctrl={control_cost:.2f}, "
                      f"coll={collision_cost:.1f}, bound={boundary_cost:.1f}, "
                      f"prox={proximity_cost:.1f}, goal={goal_cost:.1f}, term={terminal_cost:.1f}", flush=True)
                print(f"  Total: {total:.1f}", flush=True)
        
        return total

# Cost function setup
# State: [x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r] (12-DOF)
# Penalize position errors most, then velocities, then attitudes, then rates
Q_diag = np.array([
    10.0, 10.0, 10.0,  # Position errors (x, y, z)
    1.0, 1.0, 1.0,      # Velocity errors (vx, vy, vz)
    2.0, 2.0, 2.0,      # Attitude errors (roll, pitch, yaw)
    0.5, 0.5, 0.5       # Angular rate errors (p, q, r)
])
Q = np.diag(Q_diag)

# Control input penalties (4 rotor thrusts, normalized 0-1)
# Keep control effort moderate
R_diag = np.array([0.1, 0.1, 0.1, 0.1])
R = np.diag(R_diag)

# Instantiate cost function with tuned weights
cost_function = QuadSlitCostFunction(
    goal_state=x_goal,
    goal_position=goal_position,
    Q=Q,
    R=R,
    collision_weight=100.0,       # Moderate base, exponential scaling makes deep collisions expensive
    boundary_weight=10000.0,      # Very heavy penalty - absolutely avoid leaving bounds
    proximity_weight=2.0,         # Light weight for log barrier (log grows fast near obstacles)
    proximity_threshold=0.3,      # Start penalizing within 0.3m of obstacles
    goal_weight=10.0,             # Light penalty for distance from goal during trajectory
    terminal_goal_weight=5000.0,  # Dominant terminal cost - reaching goal is most important at end
    debug=True                     # Enable cost function debug output
)


# Visualization of the wall + slit configuration
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_vehicle_box(ax, vehicle, color='green', alpha=0.6):
	"""Helper to plot vehicle as a box at its current position."""
	transform = vehicle.get_transform()
	R = transform.getRotation()
	t = transform.getTranslation()
	
	# Get box half-extents for plotting (FCL stores full dimensions, we need half for vertices)
	hx, hy, hz = 0.15, 0.15, 0.05  # vehicle is 0.3x0.3x0.1, so half is 0.15x0.15x0.05
	
	# Define box vertices in local frame
	vertices = np.array([
		[-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
		[-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz]
	])
	
	# Transform vertices to world frame
	vertices_world = (R @ vertices.T).T + t
	
	# Define the 6 faces of the box
	faces = [
		[vertices_world[0], vertices_world[1], vertices_world[2], vertices_world[3]],
		[vertices_world[4], vertices_world[5], vertices_world[6], vertices_world[7]],
		[vertices_world[0], vertices_world[1], vertices_world[5], vertices_world[4]],
		[vertices_world[2], vertices_world[3], vertices_world[7], vertices_world[6]],
		[vertices_world[0], vertices_world[3], vertices_world[7], vertices_world[4]],
		[vertices_world[1], vertices_world[2], vertices_world[6], vertices_world[5]]
	]
	
	ax.add_collection3d(Poly3DCollection(faces, facecolors=color, linewidths=1, 
										 edgecolors='black', alpha=alpha))

# Create multi-angle view
fig = plt.figure(figsize=(18, 6))
	
# Calculate aspect ratios based on data ranges
x_range = dim[1] - dim[0]  # 15
y_range = dim[3] - dim[2]  # 8
z_range = abs(dim[5] - dim[4])  # 15

# View 1: Default angle
ax1 = fig.add_subplot(131, projection="3d")
config_space.plot_configuration_space(ax=ax1, obstacle_alpha=0.5,
									  title="View 1: Default with Vehicle")
plot_vehicle_box(ax1, vehicle)
ax1.scatter(*goal_position, c='green', s=100, marker='*', label='Goal')
ax1.view_init(elev=20, azim=45)
ax1.set_box_aspect([x_range, y_range, z_range])
ax1.legend()

# View 2: Front view (looking through slit)
ax2 = fig.add_subplot(132, projection="3d")
config_space.plot_configuration_space(ax=ax2, obstacle_alpha=0.5,
									  title="View 2: Front (through slit)")
plot_vehicle_box(ax2, vehicle)
ax2.scatter(*goal_position, c='green', s=100, marker='*', label='Goal')
ax2.view_init(elev=0, azim=0)
ax2.set_box_aspect([x_range, y_range, z_range])
ax2.legend()

# View 3: Side view
ax3 = fig.add_subplot(133, projection="3d")
config_space.plot_configuration_space(ax=ax3, obstacle_alpha=0.5,
									  title="View 3: Side")
plot_vehicle_box(ax3, vehicle)
ax3.scatter(*goal_position, c='green', s=100, marker='*', label='Goal')
ax3.view_init(elev=0, azim=90)
ax3.set_box_aspect([x_range, y_range, z_range])
ax3.legend()

out_path = Path(__file__).parent / "quad_slit_config.png"
plt.tight_layout()
plt.savefig(out_path, dpi=150)
plt.close(fig)
print(f"Saved slit configuration figure (3 views) with vehicle at x0 to {out_path}")

# ============================================================
# RUN MPC OPTIMIZATION
# ============================================================
print("\n" + "="*60)
print("RUNNING NAC-MPC OPTIMIZATION")
print("="*60)

# Create CEM optimizer
optimizer = CrossEntropyMethod(
    population_size=50,       # Reduced for speed (was 200)
    elite_frac=0.2,           # Keep top 20% as elites (10 samples)
    max_iterations=15,        # Reduced iterations (was 25)
    epsilon=1e-3,             # Convergence threshold
    alpha=0.0,                # No smoothing for MPC (fast adaptation)
    initial_std=0.3,          # Moderate initial exploration
    min_std=1e-3,             # Prevent premature convergence
    bounds=None,              # Use default [0, 1] bounds
    verbose=True              # Print progress
)

# Create NAC-MPC controller with CEM optimizer
from Controls.NACMPC import SplineInterpolationInput

# Initial decision vector
# Format: [horizon, kf0_u0, kf0_u1, kf0_u2, kf0_u3, kf1_u0, ...]
num_keyframes = 5
horizon = 3.0  # seconds

# Create input function instance
input_func = SplineInterpolationInput(
    numKeyframes=num_keyframes,
    totalSteps=1000,  # Will be updated during rollout
    control_dim=4,
    u_min=0.0,
    u_max=1.0
)

controller = NACMPC(
    vehicle=vehicle,
    costFunction=cost_function,
    optimizer=optimizer,
    inputFunction=input_func,
    control_dim=4,  # 4 rotor inputs
    debug=True,  # Enable debug prints
    physicsSteps=50,  # Reduced for speed (was 1000) - still adequate for control
    numControlKeyframes=num_keyframes,
    dynamicHorizon=True,
    maxHorizon=10.0,
    minHorizon=1.0
)

# Start with hover thrust for all keyframes
hover_thrust = 0.25  # Normalized hover thrust
initial_decision = np.zeros(1 + num_keyframes * 4)
initial_decision[0] = horizon
initial_decision[1:] = hover_thrust  # All rotors at hover

print(f"\nInitial guess:")
print(f"  Horizon: {horizon:.2f}s")
print(f"  Keyframes: {num_keyframes}")
print(f"  Initial control: {hover_thrust} (hover thrust)")

# Cost function context (collision checks, etc.)
cost_context = {
    'goal_state': x_goal,
    'goal_position': goal_position,
    'Q': Q,
    'R': R,
    'config_space': config_space,  # Pass config_space for collision queries
}

# Run optimization
print(f"\nStarting optimization from x0 = {x0[:3]}")
print(f"Goal position: {goal_position}")
print(f"Decision vector dimension: {len(initial_decision)}")

optimized_decision = controller.plan(x0, initial_decision, **cost_context)

print("\n" + "="*60)
print("OPTIMIZATION COMPLETE")
print("="*60)

# Extract optimized parameters
opt_horizon = optimized_decision[0]
opt_keyframe_values = optimized_decision[1:].reshape(num_keyframes, 4)

print(f"\nOptimized horizon: {opt_horizon:.3f}s")
print(f"Optimized keyframe controls:")
for i, kf in enumerate(opt_keyframe_values):
    print(f"  KF{i}: {kf}")

# Evaluate final cost
final_cost = controller.evaluate_decision_vector(optimized_decision)
initial_cost = controller.evaluate_decision_vector(initial_decision)

print(f"\nInitial cost: {initial_cost:.4f}")
print(f"Final cost: {final_cost:.4f}")
print(f"Improvement: {initial_cost - final_cost:.4f} ({100*(initial_cost-final_cost)/initial_cost:.1f}%)")

# Get convergence history
history = optimizer.get_iteration_history()
print(f"\nConvergence history ({len(history)} iterations):")
print(f"  {history}")

print("\n MPC optimization complete!")