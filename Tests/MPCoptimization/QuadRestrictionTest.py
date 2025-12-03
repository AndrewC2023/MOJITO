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

from gncpy.dynamics.aircraft import SimpleMultirotorQuat
from ConfigurationSpace.ConfigSpace3D import ConfigurationSpace3D # type: ignore
from ConfigurationSpace.Obstacles import StaticObstacle # type: ignore
from Vehicles.Vehicle import Vehicle # type: ignore
from Controls.NACMPC import NACMPC # type: ignore
from Optimizers.OptimizerBase import OptimizerBase, CostFunction # type: ignore

# configuration setup (Note we are in NED frame - this makes dynamics frame conversion easier for me)
dim = [0, 10, -2.5, 2.5, 0, -5]

GRAVITY = np.array([0,0,9.81])  # acceleration vector in NED frame

# construct the dynamics class
config_file = Path(__file__).parent / "SmallQuadrotor.yaml"
QuadDynamics = SimpleMultirotorQuat(str(config_file), effector=None)


# Vehicle geometry (0.3m x 0.3m x 0.1m)
vehicle_geometry = fcl.Box(0.3, 0.3, 0.1)

# State is 12 Dof (pos, vel, euler angles, angular rates) maybe the mixed representaiton idk
x0 = np.array([1.5, -1.25, -1.5, 0,0,0, 0,0,0, 0,0,0])  # initial state
u0 = np.array([0,0,0,0])  # initial input This actually need to be the hover

# Goal position (on the other side of the slit)
goal_position = np.array([8.5, 1.25, -3.5])  # [x, y, z] in NED

x_goal = np.array([8.5, 4, 1.5, 0,0,0, 0,0,0, 0,0,0])  # goal state

# Construct vehicle
vehicle = Vehicle(
    dynamics_class=QuadDynamics,
    geometry=vehicle_geometry,
    initial_state=x0,
    state_indices={'position': [0, 1, 2]}  # NED position in state
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

# Left side of slit (negative Y side) - full height wall
left_geom = fcl.Box(wall_thickness, upper_y_height, total_z)
left_tf = fcl.Transform(np.eye(3), [x_center, dim[2] + 0.5 * upper_y_height, z_mid])
config_space.add_obstacle(StaticObstacle(left_geom, left_tf))

# Right side of slit (positive Y side) - full height wall
right_geom = fcl.Box(wall_thickness, upper_y_height, total_z)
right_tf = fcl.Transform(np.eye(3), [x_center, dim[3] - 0.5 * upper_y_height, z_mid])
config_space.add_obstacle(StaticObstacle(right_geom, right_tf))

# Bottom of slit (across slit width at lower Z region)
bottom_geom = fcl.Box(wall_thickness, slit_width, lower_z_height)
bottom_tf = fcl.Transform(np.eye(3), [x_center, y_mid, z_mid - 0.5 * slit_height - 0.5 * lower_z_height])
config_space.add_obstacle(StaticObstacle(bottom_geom, bottom_tf))

# Top of slit (across slit width at upper Z region)
top_geom = fcl.Box(wall_thickness, slit_width, upper_z_height)
top_tf = fcl.Transform(np.eye(3), [x_center, y_mid, z_mid + 0.5 * slit_height + 0.5 * upper_z_height])
config_space.add_obstacle(StaticObstacle(top_geom, top_tf))

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
                 terminal_goal_weight: float = 500.0):
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

    def evaluate(self, state: np.ndarray, control: np.ndarray, dt: float, **kwargs) -> float:
        """Evaluate instantaneous cost for quadcopter navigation.
        
        Parameters
        ----------
        state : np.ndarray
            Current state [pos, vel, euler, rates] (12-DOF)
        control : np.ndarray
            Control inputs (normalized 0-1 as per project convention)
        dt : float
            Time step duration
        **kwargs : dict
            Must include:
            - collision_result: CollisionQueryResult from ConfigSpace3D.query_collision_detailed()
            - is_terminal: bool indicating if this is the final step
            
        Returns
        -------
        float
            Total instantaneous cost
        """
        cost = 0.0
        
        # Extract collision data from kwargs
        collision_result = kwargs.get('collision_result', None)
        is_terminal = kwargs.get('is_terminal', False)
        
        if collision_result is None:
            raise ValueError("collision_result must be provided in kwargs")
        
        # 1. LQR-style state penalty: (x - x_goal)^T Q (x - x_goal)
        state_error = state - self.goal_state
        state_cost = state_error.T @ self.Q @ state_error
        cost += state_cost
        
        # 2. LQR-style control penalty: u^T R u
        # (Assumes hover control is near zero; otherwise penalize deviation from u_hover)
        control_cost = control.T @ self.R @ control
        cost += control_cost
        
        # 3. Collision penalty (strongly penalize based on penetration depth)
        # Use exponential scaling so deeper collisions are much more expensive
        if collision_result.has_collision and collision_result.num_collisions > 0:
            # Exponential penalty: grows with penetration depth (scaled by factor of 3)
            collision_cost = self.collision_weight * (np.exp(collision_result.total_penetration_depth * 3.0) - 1.0)
            cost += collision_cost
        
        # 4. Boundary violation penalty (very severe - we never want to leave bounds)
        if collision_result.is_out_of_bounds:
            cost += self.boundary_weight
        
        # 5. Proximity penalty (logarithmic barrier to encourage safe distances)
        # Only apply if NOT already in collision (collision penalty dominates)
        # Log-law provides smooth gradient far away, steep gradient close to obstacles
        # Allows slit navigation while maintaining safe clearance
        if not collision_result.has_collision and collision_result.min_obstacle_distance < self.proximity_threshold:
            # Prevent log(0) or log(negative) - clamp minimum distance
            safe_distance = max(collision_result.min_obstacle_distance, 0.01)
            # Log barrier: -log(d) grows rapidly as d -> 0
            proximity_cost = self.proximity_weight * (-np.log(safe_distance / self.proximity_threshold))
            cost += proximity_cost
        
        # 6. Goal position tracking penalty
        current_position = state[0:3]  # Extract position from state
        goal_distance = np.linalg.norm(current_position - self.goal_position)
        goal_cost = self.goal_weight * goal_distance ** 2
        cost += goal_cost
        
        # 7. Terminal cost (large penalty if final state is far from goal)
        if is_terminal:
            terminal_cost = self.terminal_goal_weight * goal_distance ** 2
            cost += terminal_cost
        
        # Scale by time step for time-consistent cost accumulation
        # This ensures the total cost is independent of dt choice
        return cost * dt

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
    terminal_goal_weight=5000.0   # Dominant terminal cost - reaching goal is most important at end
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