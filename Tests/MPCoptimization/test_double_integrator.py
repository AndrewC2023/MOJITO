"""
Double Integrator 3D - NACMPC Framework Validation

Validates NACMPC with stable 3D point mass dynamics and collision avoidance.
Demonstrates smooth spline control navigating around obstacles.
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import fcl
import sys
from pathlib import Path

# Add project source to path
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir / "src"))

from ConfigurationSpace.ConfigSpace3D import ConfigurationSpace3D
from ConfigurationSpace.Obstacles import StaticObstacle
from Vehicles.Vehicle import Vehicle
from Controls.NACMPC import NACMPC
from Optimizers.OptimizerBase import CostFunction
from Optimizers.CrossEntropyMethod import CrossEntropyMethod
from Controls.NACMPC import SplineInterpolationInput

print("="*80)
print("DOUBLE INTEGRATOR 3D - NACMPC VALIDATION")
print("="*80)

# ============================================================================
# Dynamics
# ============================================================================

class DoubleIntegrator3D:
    """Simple 3D point mass with acceleration control.
    
    State: [x, y, z, vx, vy, vz]  (6D)
    Control: [ax, ay, az]  (3D)
    """
    
    def __init__(self, dt=0.1, max_accel=5.0):
        self.dt = dt
        self.max_accel = max_accel
        self.state = np.zeros((6, 1))
        self.control_dim = 3
    
    def set_state(self, state):
        self.state = state.reshape((-1, 1))
    
    def propagate_state(self, timestep, state, u, **kwargs):
        """Propagate state forward using Euler integration."""
        state = state.reshape((-1, 1))
        u = u.reshape((-1, 1))
        u = np.clip(u, -self.max_accel, self.max_accel)
        
        pos = state[0:3]
        vel = state[3:6]
        
        # Use the passed timestep parameter, not self.dt
        pos_next = pos + vel * timestep
        vel_next = vel + u * timestep
        
        return np.vstack([pos_next, vel_next]).reshape((-1, 1))

# ============================================================================
# Configuration Space Setup
# ============================================================================

dim = [0, 10, -2.5, 2.5, -5, 0]  # NED frame
config_space = ConfigurationSpace3D(dim)

# Add obstacles for S-curve navigation
obstacle1_box = fcl.Box(1.0, 1.0, 1.0)
obstacle1_transform = fcl.Transform(np.array([3.5, 0.0, -1.8]))
obstacle1 = StaticObstacle(obstacle1_box, obstacle1_transform)
config_space.add_obstacle(obstacle1)

obstacle2_box = fcl.Box(0.9, 0.9, 0.9)
obstacle2_transform = fcl.Transform(np.array([6.0, 0.0, -2.3]))
obstacle2 = StaticObstacle(obstacle2_box, obstacle2_transform)
config_space.add_obstacle(obstacle2)

print(f"Configuration space: {dim}")
print(f"Obstacles: {len(config_space.obstacles)}")

# ============================================================================
# Problem Setup
# ============================================================================

physics_dt = 0.1  # Define early for use in dynamics and later calculations
dynamics = DoubleIntegrator3D(dt=physics_dt, max_accel=5.0)

INIT_POS = np.array([1.0, 0.0, -1.0])
INIT_VEL = np.array([0.0, 0.0, 0.0])
x0 = np.vstack([INIT_POS.reshape(-1, 1), INIT_VEL.reshape(-1, 1)])
dynamics.set_state(x0)

goal_position = np.array([8.0, 0.0, -3.0])
goal_velocity = np.array([0.0, 0.0, 0.0])
x_goal = np.vstack([goal_position.reshape(-1, 1), goal_velocity.reshape(-1, 1)])

print(f"Start: {INIT_POS}, Goal: {goal_position}")
print(f"Distance: {np.linalg.norm(goal_position - INIT_POS):.2f}m")

# ============================================================================
# Vehicle Class Construction
# ============================================================================

vehicle_geometry = fcl.Box(0.3, 0.3, 0.3)
vehicle = Vehicle(
    dynamics_class=dynamics,
    geometry=vehicle_geometry,
    initial_state=x0,
    state_indices={'position': [0, 1, 2]}
)

# ============================================================================
# Cost Function - need to define it but arbitrary cost is the idea for this project
# ============================================================================

class PointMassCost(CostFunction):
    """LQR-style cost with smooth collision avoidance."""
    
    def __init__(self, goal_position, goal_velocity,
                 pos_weight=100.0, vel_weight=10.0, control_weight=0.1,
                 terminal_pos_weight=500.0, terminal_vel_weight=50.0,
                 terminal_control_weight=100.0,
                 collision_weight=10000.0, proximity_weight=50.0):
        super().__init__()
        self.goal_position = goal_position.reshape(-1, 1)
        self.goal_velocity = goal_velocity.reshape(-1, 1)
        
        self.pos_weight = pos_weight
        self.vel_weight = vel_weight
        self.control_weight = control_weight
        self.terminal_pos_weight = terminal_pos_weight
        self.terminal_vel_weight = terminal_vel_weight
        self.terminal_control_weight = terminal_control_weight
        self.collision_weight = collision_weight
        self.proximity_weight = proximity_weight
        
        self.best_cost = float('inf')
        self.best_state = None
    
    def reset_for_new_trajectory(self):
        """Reset internal state for new trajectory evaluation."""
        pass  # No prev_control tracking in this cost function
    
    def evaluate(self, state, control=None, is_terminal=False, dt=0.1, **kwargs):
        """Evaluate cost with dt-normalized running costs."""
        cost = 0.0
        
        pos = state[0:3].reshape(-1, 1)
        vel = state[3:6].reshape(-1, 1)
        
        # Running costs (integrated over time)
        pos_error = pos - self.goal_position
        cost += self.pos_weight * np.dot(pos_error.flatten(), pos_error.flatten()) * dt
        
        vel_error = vel - self.goal_velocity
        cost += self.vel_weight * np.dot(vel_error.flatten(), vel_error.flatten()) * dt
        
        if control is not None:
            control = control.reshape(-1, 1)
            cost += self.control_weight * np.dot(control.flatten(), control.flatten()) * dt
        
        # Collision costs (integrated over time)
        if 'collision_result' in kwargs:
            collision_result = kwargs['collision_result']
            
            if collision_result.has_collision:
                if collision_result.is_out_of_bounds:
                    cost += self.collision_weight * 2.0 * dt
                else:
                    cost += self.collision_weight * (1.0 + collision_result.total_penetration_depth) * dt
            
            # Proximity gradient for smooth optimization (integrated over time)
            if not collision_result.has_collision and collision_result.min_obstacle_distance < float('inf'):
                safe_distance = 0.5
                if collision_result.min_obstacle_distance < safe_distance:
                    proximity_factor = (safe_distance - collision_result.min_obstacle_distance) / safe_distance
                    cost += self.proximity_weight * (proximity_factor ** 2) * dt
        
        # Terminal costs
        if is_terminal:
            terminal_pos_cost = self.terminal_pos_weight * np.dot(pos_error.flatten(), pos_error.flatten())
            terminal_vel_cost = self.terminal_vel_weight * np.dot(vel_error.flatten(), vel_error.flatten())
            cost += terminal_pos_cost + terminal_vel_cost
            
            # Terminal control penalty: encourage u0 at end to stop accelerating
            if control is not None:
                terminal_ctrl_cost = self.terminal_control_weight * np.dot(control.flatten(), control.flatten())
                cost += terminal_ctrl_cost
            
            distance = np.linalg.norm(pos_error)
            if cost < self.best_cost:
                self.best_cost = cost
                self.best_state = {'position': pos.flatten(), 'velocity': vel.flatten(), 'distance': distance}
        
        return cost


cost_func = PointMassCost(
    goal_position, 
    goal_velocity,
    pos_weight=50.0,
    vel_weight=0.1,
    control_weight=0.1,
    terminal_pos_weight=5000.0,
    terminal_vel_weight=2000.0,
    terminal_control_weight=1.0,
    collision_weight=10000.0,
    proximity_weight=50.0
)

# ============================================================================
# Input Function
# ============================================================================

horizon = 10.0  # seconds
num_keyframes = 15
control_dim = 3

input_func = SplineInterpolationInput(
    numKeyframes=num_keyframes,
    totalSteps=int(horizon / physics_dt),  # Only needed for legacy compatibility
    control_dim=control_dim,
    u_min=-5.0,
    u_max=5.0
)
print(f"Horizon: {horizon}s, Keyframes: {num_keyframes}, Physics dt: {physics_dt}s")

# ============================================================================
# Optimizer
# ============================================================================

decision_dim = num_keyframes * control_dim
optimizer = CrossEntropyMethod(
    population_size=500,
    elite_frac=0.15,
    max_iterations=100,
    epsilon=1e-4,
    alpha=0.1,
    initial_std=2.5,
    min_std=0.1,
    bounds=(-5.0, 5.0),
    verbose=True
)
print(f"CEM Optimizer: pop=200, elites=15%, iters=100, alpha=0.05")

# ============================================================================
# NACMPC Controller
# ============================================================================

mpc = NACMPC(
    vehicle=vehicle,
    costFunction=cost_func,
    optimizer=optimizer,
    inputFunction=input_func,
    control_dim=control_dim,
    physics_dt=physics_dt,
    numControlKeyframes=num_keyframes,
    dynamicHorizon=False,
    maxHorizon=horizon,
    debug=False
)

mpc._x0 = x0.copy()
mpc.cost_context = {'vehicle': vehicle, 'config_space': config_space}

# ============================================================================
# Optimize
# ============================================================================

print("\nRunning optimization...")
print("="*80)

initial_guess = np.zeros(decision_dim)
best_decision = optimizer.optimize(initial_guess=initial_guess, controller=mpc)

print("="*80)

# ============================================================================
# Rollout and Visualization
# ============================================================================

print("\nGenerating trajectory rollout...")

# Set up input function to match MPC evaluation
input_func.updateStartAndEndTimes(0.0, horizon)
input_func.updateKeyFrameValues(best_decision)

# Rollout trajectory
trajectory_states = []
vehicle.set_state(x0.copy())
trajectory_states.append(vehicle.state.flatten().copy())

t = 0.0
rollout_steps = int(horizon / physics_dt)
for step in range(rollout_steps):
    u = input_func.calculateInput(t)
    state = vehicle.propagate(physics_dt, u=u)
    trajectory_states.append(state.flatten().copy())
    t += physics_dt

trajectory_states = np.array(trajectory_states)
traj_positions = trajectory_states[:, 0:3]
traj_velocities = trajectory_states[:, 3:6]
final_pos = trajectory_states[-1, 0:3]
final_vel = trajectory_states[-1, 3:6]

print(f"Final position: {final_pos}")
print(f"Final velocity: {final_vel}")

# ============================================================================
# Visualization
# ============================================================================
# Prepare obstacle data for visualization
obstacles_data = [
    {'center': np.array([3.5, 0.0, -1.8]), 'size': np.array([1.0, 1.0, 1.0])},
    {'center': np.array([6.0, 0.0, -2.3]), 'size': np.array([0.9, 0.9, 0.9])},
]

# Generate obstacle geometry for 3D plotting
all_obstacle_faces = []
for obs_data in obstacles_data:
    obs_center = obs_data['center']
    obs_size = obs_data['size']
    obs_half = obs_size / 2.0
    
    obs_corners = obs_center[:, np.newaxis] + np.array([
        [-obs_half[0], -obs_half[1], -obs_half[2]],
        [obs_half[0], -obs_half[1], -obs_half[2]],
        [obs_half[0], obs_half[1], -obs_half[2]],
        [-obs_half[0], obs_half[1], -obs_half[2]],
        [-obs_half[0], -obs_half[1], obs_half[2]],
        [obs_half[0], -obs_half[1], obs_half[2]],
        [obs_half[0], obs_half[1], obs_half[2]],
        [-obs_half[0], obs_half[1], obs_half[2]]
    ]).T
    
    faces = [
        [obs_corners[:, 0], obs_corners[:, 1], obs_corners[:, 2], obs_corners[:, 3]],
        [obs_corners[:, 4], obs_corners[:, 5], obs_corners[:, 6], obs_corners[:, 7]],
        [obs_corners[:, 0], obs_corners[:, 1], obs_corners[:, 5], obs_corners[:, 4]],
        [obs_corners[:, 2], obs_corners[:, 3], obs_corners[:, 7], obs_corners[:, 6]],
        [obs_corners[:, 0], obs_corners[:, 3], obs_corners[:, 7], obs_corners[:, 4]],
        [obs_corners[:, 1], obs_corners[:, 2], obs_corners[:, 6], obs_corners[:, 5]]
    ]
    all_obstacle_faces.append(faces)
fig = plt.figure(figsize=(18, 12))
ax1 = fig.add_subplot(221, projection='3d')  # 3D view 1
ax2 = fig.add_subplot(222, projection='3d')  # 3D view 2
ax3 = fig.add_subplot(223)  # Top-down (X-Y)
ax4 = fig.add_subplot(224)  # Side view (X-Z)

# Plot configuration space boundaries
x_bounds = [dim[0], dim[1]]
y_bounds = [dim[2], dim[3]]
z_bounds = [dim[4], dim[5]]

# Helper function to plot on 3D axes
def plot_3d_scene(ax, show_legend=True):
    # Draw boundary box (wireframe)
    from itertools import product
    corners = np.array(list(product(x_bounds, y_bounds, z_bounds)))
    # Draw edges of the box
    for i in range(2):
        for j in range(2):
            # X edges
            ax.plot([x_bounds[0], x_bounds[1]], [y_bounds[i], y_bounds[i]], [z_bounds[j], z_bounds[j]], 'k--', alpha=0.3, linewidth=0.5)
            # Y edges
            ax.plot([x_bounds[i], x_bounds[i]], [y_bounds[0], y_bounds[1]], [z_bounds[j], z_bounds[j]], 'k--', alpha=0.3, linewidth=0.5)
            # Z edges
            ax.plot([x_bounds[i], x_bounds[i]], [y_bounds[j], y_bounds[j]], [z_bounds[0], z_bounds[1]], 'k--', alpha=0.3, linewidth=0.5)
    
    # Draw all obstacles
    for faces in all_obstacle_faces:
        obstacle_collection = Poly3DCollection(faces, alpha=0.3, facecolor='red', edgecolor='darkred', linewidths=1.5)
        ax.add_collection3d(obstacle_collection)
    
    # Plot trajectory
    ax.plot(traj_positions[:, 0], traj_positions[:, 1], traj_positions[:, 2], 
            'b-', linewidth=2, label='Trajectory', alpha=0.8)
    
    # Plot start and goal
    ax.scatter(*INIT_POS, color='green', s=200, marker='o', label='Start', edgecolors='darkgreen', linewidths=2)
    ax.scatter(*goal_position, color='red', s=200, marker='*', label='Goal', edgecolors='darkred', linewidths=2)
    ax.scatter(*final_pos, color='blue', s=150, marker='x', label='Final', linewidths=3)
    
    # Draw error line
    ax.plot([final_pos[0], goal_position[0]], 
            [final_pos[1], goal_position[1]], 
            [final_pos[2], goal_position[2]], 
            'r--', alpha=0.5, linewidth=1.5, label=f'Error: {np.linalg.norm(final_pos - goal_position):.3f}m')
    
    # Labels and formatting
    ax.set_xlabel('X [m]', fontsize=10, fontweight='bold')
    ax.set_ylabel('Y [m]', fontsize=10, fontweight='bold')
    ax.set_zlabel('Z [m]', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Set axis limits
    ax.set_xlim(x_bounds)
    ax.set_ylim(y_bounds)
    ax.set_zlim(z_bounds)
    
    # Set equal aspect ratio
    ax.set_box_aspect([
        x_bounds[1] - x_bounds[0],
        y_bounds[1] - y_bounds[0],
        z_bounds[1] - z_bounds[0]
    ])
    
    if show_legend:
        ax.legend(loc='upper left', fontsize=8)

# Plot 3D views from different angles
plot_3d_scene(ax1, show_legend=True)
ax1.view_init(elev=20, azim=45)
ax1.set_title('3D View 1 (Angle 1)', fontsize=11, fontweight='bold')

plot_3d_scene(ax2, show_legend=False)
ax2.view_init(elev=10, azim=135)
ax2.set_title('3D View 2 (Angle 2)', fontsize=11, fontweight='bold')

# Top-down view (X-Y plane) - BEST for seeing obstacle avoidance
ax3.plot(traj_positions[:, 0], traj_positions[:, 1], 'b-', linewidth=2, label='Trajectory', alpha=0.8)
# Draw all obstacle footprints
for i, obs_data in enumerate(obstacles_data):
    obs_center = obs_data['center']
    obs_size = obs_data['size']
    obs_half_x = obs_size[0] / 2.0
    obs_half_y = obs_size[1] / 2.0
    obs_xy = np.array([
        [obs_center[0] - obs_half_x, obs_center[1] - obs_half_y],
        [obs_center[0] + obs_half_x, obs_center[1] - obs_half_y],
        [obs_center[0] + obs_half_x, obs_center[1] + obs_half_y],
        [obs_center[0] - obs_half_x, obs_center[1] + obs_half_y],
        [obs_center[0] - obs_half_x, obs_center[1] - obs_half_y]
    ])
    label = 'Obstacles' if i == 0 else None
    ax3.fill(obs_xy[:, 0], obs_xy[:, 1], color='red', alpha=0.25, edgecolor='darkred', linewidth=1.5, label=label)
ax3.scatter(*INIT_POS[0:2], color='green', s=200, marker='o', label='Start', edgecolors='darkgreen', linewidths=2, zorder=5)
ax3.scatter(*goal_position[0:2], color='red', s=200, marker='*', label='Goal', edgecolors='darkred', linewidths=2, zorder=5)
ax3.scatter(*final_pos[0:2], color='blue', s=150, marker='x', label='Final', linewidths=3, zorder=5)
ax3.plot([final_pos[0], goal_position[0]], [final_pos[1], goal_position[1]], 'r--', alpha=0.5, linewidth=1.5)
ax3.set_xlabel('X [m]', fontsize=10, fontweight='bold')
ax3.set_ylabel('Y [m]', fontsize=10, fontweight='bold')
ax3.set_title('Top-Down View (X-Y)', fontsize=11, fontweight='bold')
ax3.set_xlim(x_bounds)
ax3.set_ylim(y_bounds)
ax3.set_aspect('equal')
ax3.grid(True, alpha=0.3)
ax3.legend(loc='upper left', fontsize=8)

# Side view (X-Z plane)
ax4.plot(traj_positions[:, 0], traj_positions[:, 2], 'b-', linewidth=2, label='Trajectory', alpha=0.8)
for i, obs_data in enumerate(obstacles_data):
    obs_center = obs_data['center']
    obs_size = obs_data['size']
    obs_half_x = obs_size[0] / 2.0
    obs_half_z = obs_size[2] / 2.0
    obs_xz = np.array([
        [obs_center[0] - obs_half_x, obs_center[2] - obs_half_z],
        [obs_center[0] + obs_half_x, obs_center[2] - obs_half_z],
        [obs_center[0] + obs_half_x, obs_center[2] + obs_half_z],
        [obs_center[0] - obs_half_x, obs_center[2] + obs_half_z],
        [obs_center[0] - obs_half_x, obs_center[2] - obs_half_z]
    ])
    label = 'Obstacles' if i == 0 else None
    ax4.fill(obs_xz[:, 0], obs_xz[:, 1], color='red', alpha=0.25, edgecolor='darkred', linewidth=1.5, label=label)
ax4.scatter(INIT_POS[0], INIT_POS[2], color='green', s=200, marker='o', label='Start', edgecolors='darkgreen', linewidths=2, zorder=5)
ax4.scatter(goal_position[0], goal_position[2], color='red', s=200, marker='*', label='Goal', edgecolors='darkred', linewidths=2, zorder=5)
ax4.scatter(final_pos[0], final_pos[2], color='blue', s=150, marker='x', label='Final', linewidths=3, zorder=5)
ax4.plot([final_pos[0], goal_position[0]], [final_pos[2], goal_position[2]], 'r--', alpha=0.5, linewidth=1.5)
ax4.set_xlabel('X [m]', fontsize=10, fontweight='bold')
ax4.set_ylabel('Z [m]', fontsize=10, fontweight='bold')
ax4.set_title('Side View (X-Z)', fontsize=11, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.set_xlim(x_bounds)
ax4.set_ylim(z_bounds)
ax4.set_aspect('equal')
ax4.legend(loc='upper left', fontsize=8)

fig.suptitle('Double Integrator - 3D Views and Projections', fontsize=14, fontweight='bold', y=0.98)

output_path = Path(__file__).parent / "double_integrator_3d_views.png"
plt.tight_layout()
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n3D views saved: {output_path}")
plt.close()

# ============================================================================
# Visualization - Goal Tracking
# ============================================================================

fig2 = plt.figure(figsize=(16, 10))

# Position vs time with goal lines
ax1 = fig2.add_subplot(221)
time_vec = np.linspace(0, horizon, len(traj_positions))
ax1.plot(time_vec, traj_positions[:, 0], 'b-', linewidth=2, label='X', alpha=0.8)
ax1.plot(time_vec, traj_positions[:, 1], 'g-', linewidth=2, label='Y', alpha=0.8)
ax1.plot(time_vec, traj_positions[:, 2], 'r-', linewidth=2, label='Z', alpha=0.8)
ax1.axhline(goal_position[0], color='b', linestyle='--', alpha=0.4, linewidth=1.5, label='X goal')
ax1.axhline(goal_position[1], color='g', linestyle='--', alpha=0.4, linewidth=1.5, label='Y goal')
ax1.axhline(goal_position[2], color='r', linestyle='--', alpha=0.4, linewidth=1.5, label='Z goal')
ax1.set_xlabel('Time [s]', fontsize=10, fontweight='bold')
ax1.set_ylabel('Position [m]', fontsize=10, fontweight='bold')
ax1.set_title('Position vs Time (Goal Tracking)', fontsize=11, fontweight='bold')
ax1.legend(loc='best', fontsize=8, ncol=2)
ax1.grid(True, alpha=0.3)

# Velocity vs time
ax2 = fig2.add_subplot(222)
ax2.plot(time_vec, traj_velocities[:, 0], 'b-', linewidth=2, label='Vx', alpha=0.8)
ax2.plot(time_vec, traj_velocities[:, 1], 'g-', linewidth=2, label='Vy', alpha=0.8)
ax2.plot(time_vec, traj_velocities[:, 2], 'r-', linewidth=2, label='Vz', alpha=0.8)
ax2.axhline(0, color='k', linestyle='--', alpha=0.3, linewidth=1)
ax2.set_xlabel('Time [s]', fontsize=10, fontweight='bold')
ax2.set_ylabel('Velocity [m/s]', fontsize=10, fontweight='bold')
ax2.set_title('Velocity vs Time', fontsize=11, fontweight='bold')
ax2.legend(loc='best', fontsize=8)
ax2.grid(True, alpha=0.3)

# Position error magnitude
ax3 = fig2.add_subplot(223)
pos_error_mag = np.linalg.norm(traj_positions - goal_position, axis=1)
ax3.plot(time_vec, pos_error_mag, 'r-', linewidth=2, alpha=0.8)
ax3.axhline(0.5, color='orange', linestyle='--', alpha=0.5, linewidth=1.5, label='Success threshold (0.5m)')
ax3.set_xlabel('Time [s]', fontsize=10, fontweight='bold')
ax3.set_ylabel('Position Error [m]', fontsize=10, fontweight='bold')
ax3.set_title('Position Error Magnitude', fontsize=11, fontweight='bold')
ax3.legend(loc='best', fontsize=8)
ax3.grid(True, alpha=0.3)

# Velocity magnitude
ax4 = fig2.add_subplot(224)
vel_mag = np.linalg.norm(traj_velocities, axis=1)
ax4.plot(time_vec, vel_mag, 'b-', linewidth=2, alpha=0.8)
ax4.axhline(0.5, color='orange', linestyle='--', alpha=0.5, linewidth=1.5, label='Success threshold (0.5m/s)')
ax4.set_xlabel('Time [s]', fontsize=10, fontweight='bold')
ax4.set_ylabel('Velocity Magnitude [m/s]', fontsize=10, fontweight='bold')
ax4.set_title('Velocity Magnitude', fontsize=11, fontweight='bold')
ax4.legend(loc='best', fontsize=8)
ax4.grid(True, alpha=0.3)

fig2.suptitle('Double Integrator - Goal Tracking Performance', fontsize=14, fontweight='bold', y=0.98)

output_path2 = Path(__file__).parent / "double_integrator_tracking.png"
plt.tight_layout()
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"Tracking plots saved: {output_path2}")
plt.close()

# ============================================================================
# Evaluate Results
# ============================================================================

if cost_func.best_state:
    final_distance = cost_func.best_state['distance']
    final_vel_mag = np.linalg.norm(cost_func.best_state['velocity'])
    
    position_tolerance = 0.5
    velocity_tolerance = 0.5
    success = (final_distance < position_tolerance and final_vel_mag < velocity_tolerance)
    
    print(f"\n{'='*80}")
    print(f"Position error: {final_distance:.4f}m (tolerance: {position_tolerance}m)")
    print(f"Velocity: {final_vel_mag:.4f}m/s (tolerance: {velocity_tolerance}m/s)")
    
    if success:
        print(f"TEST PASSED - NACMPC framework validated")
    else:
        print(f" TEST FAILED - Tolerance not met")
    print(f"{'='*80}\n")
