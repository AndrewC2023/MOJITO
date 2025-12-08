"""
Quadrotor MPC - NBBMPC Framework with LQR Inner Loop

Validates NBBMPC with a quadrotor using LQR for stabilization.
MPC plans position commands, LQR tracks them with motor control.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.linalg
import fcl
import sys
import signal
import datetime
from pathlib import Path

# Add project source to path
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir / "src"))
sys.path.append(str(root_dir / "dependencies" / "gncpy" / "src"))

from gncpy.dynamics.aircraft.simple_multirotor_quat import SimpleMultirotorQuat, v_smap_quat
import gncpy.math as gmath

from ConfigurationSpace.ConfigSpace3D import ConfigurationSpace3D
from Vehicles.Vehicle import Vehicle
from Controls.NBBMPC import NBBMPC
from Optimizers.OptimizerBase import CostFunction
from Optimizers.CrossEntropyMethod import CrossEntropyMethod
from Controls.NBBMPC import PiecewiseConstantInput  # Zero-order hold

# ============================================================================
# Interrupt Handler for Saving Current Best on Ctrl C
# ============================================================================

class OptimizationInterrupt(Exception):
    """Custom exception for graceful optimization interruption."""
    pass

interrupt_requested = False

def signal_handler(sig, frame):
    """Handle Ctrl C to save current best and exit gracefully."""
    global interrupt_requested
    print("\n\n" + "="*80)
    print("INTERRUPT DETECTED - Will save after current iteration...")
    print("="*80)
    interrupt_requested = True
    # Raise exception to break out of optimizer loop
    raise OptimizationInterrupt("User requested interrupt")

signal.signal(signal.SIGINT, signal_handler)

def plot_current_best(decision_vector, input_func, vehicle, quad_dynamics, x0, 
                      horizon, physics_dt, INIT_POS, goal_position, dim, 
                      wall_obstacles, timestamp_suffix=""):
    """Plot trajectory from current decision vector."""
    
    # Setup input function
    input_func.updateStartAndEndTimes(0.0, horizon)
    input_func.updateKeyFrameValues(decision_vector)
    
    # Rollout trajectory
    trajectory_states = []
    quad_dynamics.set_state(x0.copy())
    vehicle.set_state(x0.copy())
    trajectory_states.append(vehicle.state.flatten().copy())
    
    t = 0.0
    rollout_steps = int(horizon / physics_dt)
    for step in range(rollout_steps):
        pos_cmd = input_func.calculateInput(t)
        state = vehicle.propagate(physics_dt, u=pos_cmd)
        trajectory_states.append(state.flatten().copy())
        t += physics_dt
    
    trajectory_states = np.array(trajectory_states)
    traj_positions = trajectory_states[:, 0:3]
    final_pos = trajectory_states[-1, 0:3]
    
    print(f"Final position: {final_pos}")
    print(f"Goal position: {goal_position}")
    print(f"Position error: {np.linalg.norm(final_pos - goal_position):.4f}m")
    
    # Create visualization
    fig = plt.figure(figsize=(18, 12))
    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222, projection='3d')
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    
    x_bounds = [dim[0], dim[1]]
    y_bounds = [dim[2], dim[3]]
    z_bounds = [dim[4], dim[5]]
    
    # Helper to draw obstacles in 3D
    def draw_walls_3d(ax):
        for wall in wall_obstacles:
            center = wall['center']
            size = wall['size']
            half = size / 2.0
            corners = center[:, np.newaxis] + np.array([
                [-half[0], -half[1], -half[2]], [half[0], -half[1], -half[2]],
                [half[0], half[1], -half[2]], [-half[0], half[1], -half[2]],
                [-half[0], -half[1], half[2]], [half[0], -half[1], half[2]],
                [half[0], half[1], half[2]], [-half[0], half[1], half[2]]
            ]).T
            faces = [
                [corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]],
                [corners[:, 4], corners[:, 5], corners[:, 6], corners[:, 7]],
                [corners[:, 0], corners[:, 1], corners[:, 5], corners[:, 4]],
                [corners[:, 2], corners[:, 3], corners[:, 7], corners[:, 6]],
                [corners[:, 0], corners[:, 3], corners[:, 7], corners[:, 4]],
                [corners[:, 1], corners[:, 2], corners[:, 6], corners[:, 5]]
            ]
            ax.add_collection3d(Poly3DCollection(faces, alpha=0.3, facecolor='gray', edgecolor='black', linewidth=0.5))
    
    # Plot 3D views
    for ax, elev, azim, title in [(ax1, 20, 45, '3D View 1'), (ax2, 10, 135, '3D View 2')]:
        ax.plot(traj_positions[:, 0], traj_positions[:, 1], traj_positions[:, 2], 
                'b-', linewidth=2, label='Trajectory', alpha=0.8)
        draw_walls_3d(ax)
        ax.scatter(*INIT_POS, color='green', s=200, marker='o', label='Start', edgecolors='darkgreen', linewidths=2, zorder=5)
        ax.scatter(*goal_position, color='red', s=200, marker='*', label='Goal', edgecolors='darkred', linewidths=2, zorder=5)
        ax.scatter(*final_pos, color='blue', s=150, marker='x', label='Final', linewidths=3, zorder=5)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.set_xlim(x_bounds)
        ax.set_ylim(y_bounds)
        ax.set_zlim(z_bounds)
        ax.set_title(title, fontweight='bold')
        ax.view_init(elev=elev, azim=azim)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Top-down view (XY)
    ax3.plot(traj_positions[:, 0], traj_positions[:, 1], 'b-', linewidth=2, label='Trajectory', alpha=0.8)
    for wall in wall_obstacles:
        center, size = wall['center'], wall['size']
        rect = plt.Rectangle((center[0]-size[0]/2, center[1]-size[1]/2), size[0], size[1],
                             facecolor='gray', alpha=0.3, edgecolor='black', linewidth=1)
        ax3.add_patch(rect)
    ax3.scatter(*INIT_POS[0:2], color='green', s=200, marker='o', label='Start', edgecolors='darkgreen', linewidths=2, zorder=5)
    ax3.scatter(*goal_position[0:2], color='red', s=200, marker='*', label='Goal', edgecolors='darkred', linewidths=2, zorder=5)
    ax3.scatter(*final_pos[0:2], color='blue', s=150, marker='x', label='Final', linewidths=3, zorder=5)
    ax3.set_xlabel('X [m]', fontweight='bold')
    ax3.set_ylabel('Y [m]', fontweight='bold')
    ax3.set_title('Top-Down View (X-Y)', fontweight='bold')
    ax3.set_xlim(x_bounds)
    ax3.set_ylim(y_bounds)
    ax3.set_aspect('equal')
    ax3.legend(loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Side view (XZ)
    ax4.plot(traj_positions[:, 0], traj_positions[:, 2], 'b-', linewidth=2, label='Trajectory', alpha=0.8)
    for wall in wall_obstacles:
        center, size = wall['center'], wall['size']
        bottom_left_x = center[0] - size[0]/2
        bottom_left_z = center[2] - size[2]/2
        rect = plt.Rectangle((bottom_left_x, bottom_left_z), size[0], size[2],
                             facecolor='gray', alpha=0.3, edgecolor='black', linewidth=1)
        ax4.add_patch(rect)
    ax4.scatter(INIT_POS[0], INIT_POS[2], color='green', s=200, marker='o', label='Start', edgecolors='darkgreen', linewidths=2, zorder=5)
    ax4.scatter(goal_position[0], goal_position[2], color='red', s=200, marker='*', label='Goal', edgecolors='darkred', linewidths=2, zorder=5)
    ax4.scatter(final_pos[0], final_pos[2], color='blue', s=150, marker='x', label='Final', linewidths=3, zorder=5)
    ax4.set_xlabel('X [m]', fontweight='bold')
    ax4.set_ylabel('Z [m]', fontweight='bold')
    ax4.set_title('Side View (X-Z)', fontweight='bold')
    ax4.set_xlim(x_bounds)
    ax4.set_ylim(z_bounds)
    ax4.set_aspect('equal')
    ax4.legend(loc='upper left', fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    fig.suptitle(f'Quadrotor MPC - {timestamp_suffix}', fontsize=14, fontweight='bold', y=0.98)
    output_path = Path(__file__).parent / f"quadrotor_mpc_snapshot_{timestamp_suffix}.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSnapshot saved: {output_path}")
    plt.close()
    
    return final_pos, traj_positions

print("="*80)
print("QUADROTOR MPC - LQR INNER LOOP + MPC OUTER LOOP")
print("="*80)
print("\nPress Ctrl+C at any time to save current best trajectory and exit")
print("="*80)

# ============================================================================
# Controller + Quadrotor Wrapper to manage vehicle interface
# ============================================================================

class PositionControlledQuadrotor:
    """Wrapper combining SimpleMultirotorQuat and a LQR controller.
    
    This creates a "position-controlled quadrotor" where:
    - Control input: 3D position command [x, y, z] in NED frame
    - LQR inner loop: Stabilizes attitude/rates/velocity to track position
    - State: 12D [pos(3), vel(3), euler(3), rates(3)] for compatibility
    
    The actual vehicle uses quaternions internally, but we convert to
    Euler angles for the LQR controller to calculate Eular angle error to
    make it easier for a user to command specific attitudes.

    This wrapper is based heavily in the gncpy SimpleMultirotor validation test
    """
    
    def __init__(self, config_file, dt=0.01):
        """Initialize quadrotor with LQR controller.
        
        Parameters
        ----------
        config_file : str or Path
            Path to quadrotor YAML configuration
        dt : float
            Control timestep in seconds
        """
        self.dt = dt
        self.gravity = 9.81
        
        # Initialize quadrotor
        self.quad = SimpleMultirotorQuat(str(config_file), effector=None)
        
        # Set initial conditions (hover at origin)
        ned_mag = np.array([20.0, 5.0, 45.0])
        init_pos = np.array([0.0, 0.0, -2.0])
        init_vel = np.array([0.0, 0.0, 0.0])
        init_euler = np.array([0.0, 0.0, 0.0])
        init_rates = np.array([0.0, 0.0, 0.0])
        
        self.quad.set_initial_conditions(
            init_pos, init_vel, init_euler, init_rates,
            34.0, -86.0, 0.0, ned_mag
        )
        self.quad.vehicle.takenoff = True
        
        # Store reference conditions for proper reinitialization
        self.ref_lat = 34.0
        self.ref_lon = -86.0
        self.terrain_alt = 0.0
        self.ned_mag = ned_mag
        
        # Calculate hover motor command
        self.mass = self.quad.vehicle.params.mass.mass_kg
        hover_thrust_per_motor = self.mass * self.gravity / self.quad.vehicle.params.motor.num_motors
        
        thrust_poly = np.polynomial.Polynomial(self.quad.vehicle.params.prop.poly_thrust[-1::-1])
        roots = (thrust_poly - hover_thrust_per_motor).roots()
        valid_roots = roots[np.isreal(roots) & (roots.real > 0) & (roots.real < 1)]
        self.hover_cmd = float(valid_roots[0].real) if len(valid_roots) > 0 else 0.7
        self.trim_motor_cmd = np.ones(4) * self.hover_cmd
        
        print(f"Quadrotor initialized: mass={self.mass:.3f}kg, hover_cmd={self.hover_cmd:.3f}")
        
        # Design LQR controller
        self._design_lqr()
        
        # State dimension for Vehicle class (12D Euler representation)
        self.state = self._get_euler_state()
        self.control_dim = 3  # Position commands [x, y, z]
    
    def _design_lqr(self):
        """Design LQR controller for attitude/rate stabilization."""
        print("Designing LQR controller...")
        
        # Define state derivative function for linearization
        def state_deriv(x, u, t):
            """12-DOF dynamics: [ned_pos(3), body_vel(3), euler(3), body_rates(3)]."""
            old_state = self.quad.vehicle.state.copy()
            
            roll, pitch, yaw = x[6:9]
            quat = gmath.euler_to_quat(roll, pitch, yaw)
            
            self.quad.vehicle.state[v_smap_quat.ned_pos] = x[0:3]
            self.quad.vehicle.state[v_smap_quat.body_vel] = x[3:6]
            self.quad.vehicle.state[v_smap_quat.quat] = quat
            self.quad.vehicle.state[v_smap_quat.body_rot_rate] = x[9:12]
            
            dcm_e2b = gmath.quat_to_dcm(quat)
            gravity_body = dcm_e2b @ np.array([0, 0, self.gravity * self.mass])
            prop_force, prop_mom = self.quad.vehicle._calc_prop_force_mom(u)
            force = gravity_body + prop_force
            
            ned_vel = dcm_e2b.T @ x[3:6]
            body_accel = force / self.mass + np.cross(x[3:6], x[9:12])
            
            s_phi, c_phi = np.sin(roll), np.cos(roll)
            t_theta, c_theta = np.tan(pitch), np.cos(pitch)
            c_theta = max(abs(c_theta), 1e-6) * np.sign(c_theta) if c_theta != 0 else 1e-6
            eul_dot_mat = np.array([
                [1, s_phi * t_theta, c_phi * t_theta],
                [0, c_phi, -s_phi],
                [0, s_phi / c_theta, c_phi / c_theta],
            ])
            euler_rates = eul_dot_mat @ x[9:12]

            # this does expose us to sigularities
            # TODO: consider using quaternions directly in LQR
            
            inertia = np.array(self.quad.vehicle.params.mass.inertia_kgm2)
            body_rot_accel = np.linalg.inv(inertia) @ (
                prop_mom - np.cross(x[9:12], inertia @ x[9:12])
            )
            
            self.quad.vehicle.state = old_state
            return np.concatenate([ned_vel, body_accel, euler_rates, body_rot_accel])
        
        # Get trim state
        quat_trim = self.quad.vehicle.state[v_smap_quat.quat]
        roll_trim, pitch_trim, yaw_trim = gmath.quat_to_euler(quat_trim)
        
        trim_state = np.concatenate([
            self.quad.vehicle.state[v_smap_quat.ned_pos],
            self.quad.vehicle.state[v_smap_quat.body_vel],
            np.array([roll_trim, pitch_trim, yaw_trim]),
            self.quad.vehicle.state[v_smap_quat.body_rot_rate],
        ])
        
        # Linearize
        A, B = gmath.linearize_dynamics(state_deriv, trim_state, self.trim_motor_cmd)
        
        # LQR weights - softer attitude/rate penalties for smoother control
        Q = scipy.linalg.block_diag(
            np.diag([1.0, 1.0, 1.0]),      # position tracking
            np.diag([0.1, 0.1, 0.1]),      # velocity
            np.diag([0.05, 0.05, 0.02]),   # euler angles - softer (yaw less important)
            np.diag([0.05, 0.05, 0.05]),   # rates - softer to reduce aggressiveness
        )
        R = np.diag([1.0, 1.0, 1.0, 1.0])  # motor commands
        
        # Solve for LQR gain
        P = scipy.linalg.solve_continuous_are(A, B, Q, R)
        self.K = np.linalg.inv(R) @ B.T @ P
        
        print(f"LQR gain shape: {self.K.shape}, Max gain: {np.max(np.abs(self.K)):.4f}")
    
    def _get_euler_state(self):
        """Extract 12D Euler state from quaternion-based quad state."""
        quat = self.quad.vehicle.state[v_smap_quat.quat]
        roll, pitch, yaw = gmath.quat_to_euler(quat)
        
        return np.concatenate([
            self.quad.vehicle.state[v_smap_quat.ned_pos],
            self.quad.vehicle.state[v_smap_quat.body_vel],
            np.array([roll, pitch, yaw]),
            self.quad.vehicle.state[v_smap_quat.body_rot_rate],
        ]).reshape(-1, 1)
    
    def set_state(self, state):
        """Set state (converts Euler to quaternion internally and reinitializes properly at hover)."""
        state = state.flatten()
        
        # Extract components
        pos = state[0:3]
        vel = state[3:6]
        euler = state[6:9]  # [roll, pitch, yaw] from quat_to_euler
        rates = state[9:12]
        
        # set_initial_conditions expects [yaw, pitch, roll] order in DEGREES
        # but quat_to_euler returns [roll, pitch, yaw] in RADIANS
        euler_ypr_degrees = np.array([
            np.rad2deg(euler[2]),  # yaw
            np.rad2deg(euler[1]),  # pitch
            np.rad2deg(euler[0]),  # roll
        ])
        
        self.quad.set_initial_conditions(
            pos, vel, euler_ypr_degrees, rates,
            self.ref_lat, self.ref_lon, self.terrain_alt, self.ned_mag
        )
        self.quad.vehicle.takenoff = True
        
        # Update our exposed state
        self.state = state.reshape(-1, 1)
    
    def propagate_state(self, timestep, state, u, **kwargs):
        """Propagate state with pure LQR control (like gncpy validation test).
        
        Parameters
        ----------
        timestep : float
            Time step
        state : np.ndarray
            Current 12D state [pos, vel, euler, rates]
        u : np.ndarray
            Position command [x, y, z] in NED frame
        
        Returns
        -------
        np.ndarray
            Next 12D state
        """
        # Set current state
        self.set_state(state)
        
        # Position command from MPC
        pos_cmd = u.flatten()[0:3]
        
        # Get current state in Euler representation
        x_current = self._get_euler_state().flatten()
        
        # Target state: commanded position, zero velocity/attitude/rates
        x_target = np.concatenate([
            pos_cmd,
            np.zeros(9)  # zero velocity, euler angles, rates
        ])
        
        # LQR control
        delta_u = -self.K @ (x_current - x_target)
        motor_cmd = self.trim_motor_cmd + delta_u
        motor_cmd = np.clip(motor_cmd, 0.1, 0.95)
        
        # Propagate
        self.quad.propagate_state(timestep, self.quad.vehicle.state, u=motor_cmd)
        
        # Extract new state
        new_state = self._get_euler_state()
        self.state = new_state
        
        return new_state


# ============================================================================
# Environment Setup
# ============================================================================

dim = [0, 10, -2.5, 2.5, -5, 0]  # NED frame [x_min, x_max, y_min, y_max, z_min, z_max]
config_space = ConfigurationSpace3D(dim)

# Add staggered wall obstacles
wall1_height = 2.5
wall1_box = fcl.Box(0.5, 5.0, wall1_height)
wall1_center = np.array([3.5, 0.0, -1.25])
wall1_transform = fcl.Transform(wall1_center)
from ConfigurationSpace.Obstacles import StaticObstacle
wall1 = StaticObstacle(wall1_box, wall1_transform)
config_space.add_obstacle(wall1)

wall2_height = 2.5  
wall2_box = fcl.Box(0.5, 5.0, wall2_height)
wall2_center = np.array([6.5, 0.0, -3.75]) 
wall2_transform = fcl.Transform(wall2_center)
wall2 = StaticObstacle(wall2_box, wall2_transform)
config_space.add_obstacle(wall2)

print(f"Configuration space: {dim}")
print(f"Obstacles: {len(config_space.obstacles)}")


# ============================================================================
# Dynamics and Initial Conditions
# ============================================================================

physics_dt = 0.02  # 50Hz for quadrotor control - much smoother!
config_file = Path(__file__).parent / "SmallQuadrotor.yaml"
quad_dynamics = PositionControlledQuadrotor(config_file, dt=physics_dt)

# Initial state: hover at [2, 0, -2.0]
INIT_POS = np.array([1.75, 0.0, -2.0])
INIT_VEL = np.array([0.0, 0.0, 0.0])
INIT_EULER = np.array([0.0, 0.0, 0.0])
INIT_RATES = np.array([0.0, 0.0, 0.0])

x0 = np.concatenate([INIT_POS, INIT_VEL, INIT_EULER, INIT_RATES]).reshape(-1, 1)
quad_dynamics.set_state(x0)

# Goal state
goal_position = np.array([8.0, 0.0, -2.5])
goal_velocity = np.array([0.0, 0.0, 0.0])

print(f"Start: {INIT_POS}, Goal: {goal_position}")
print(f"Distance: {np.linalg.norm(goal_position - INIT_POS):.2f}m")

# ============================================================================
# Vehicle
# ============================================================================

vehicle_geometry = fcl.Box(0.3, 0.3, 0.15)  # 30cm x 30cm x 15cm
vehicle = Vehicle(
    dynamics_class=quad_dynamics,
    geometry=vehicle_geometry,
    initial_state=x0,
    state_indices={'position': [0, 1, 2]}  # Position is first 3 elements
)

# ============================================================================
# Cost Function
# ============================================================================

class QuadrotorPositionCost(CostFunction):
    """Cost function for MPC commanding position setpoints.
    
    Penalizes position error, command changes, body rates, and Euler angles.
    LQR handles the inner loop tracking.
    """
    
    def __init__(self, goal_position,
                 pos_weight=100.0, terminal_pos_weight=500.0,
                 euler_weight=5.0,  # Penalize large Euler angles
                 rate_weight=2.0,   # Penalize large body rates
                 control_weight=0, control_change_weight=1.0,
                 collision_weight=10000.0, proximity_weight=50.0):
        super().__init__()
        self.goal_position = goal_position.reshape(-1, 1)
        
        self.pos_weight = pos_weight
        self.terminal_pos_weight = terminal_pos_weight
        self.euler_weight = euler_weight  # NEW
        self.rate_weight = rate_weight    # NEW
        self.control_weight = control_weight
        self.control_change_weight = control_change_weight
        self.collision_weight = collision_weight
        self.proximity_weight = proximity_weight
        
        self.best_cost = float('inf')
        self.best_state = None
        self.prev_control = None
        
        # Debug counters
        self.collision_count = 0
        self.eval_count = 0
        self.last_print_iter = -1
    
    def reset_for_new_trajectory(self):
        """Reset internal state for new trajectory evaluation."""
        self.prev_control = None
        self.eval_count += 1
    
    def evaluate(self, state, control=None, is_terminal=False, dt=0.01, **kwargs):
        """Evaluate cost - position, Euler angles, and body rates."""
        cost = 0.0
        collision_cost = 0.0
        
        pos = state[0:3].reshape(-1, 1)
        euler = state[6:9].reshape(-1, 1)  # NEW: extract Euler angles
        rates = state[9:12].reshape(-1, 1)  # NEW: extract body rates
        
        # Running position cost
        pos_error = pos - self.goal_position
        cost += self.pos_weight * np.dot(pos_error.flatten(), pos_error.flatten()) * dt
        
        # NEW: Euler angle penalty (want small roll/pitch, yaw doesn't matter much)
        # Weight roll and pitch more than yaw
        euler_penalty = float(euler[0]**2 + euler[1]**2 + 0.1*euler[2]**2)
        cost += self.euler_weight * euler_penalty * dt
        
        # NEW: Body rate penalty (want smooth, gentle maneuvers)
        rate_magnitude_sq = np.dot(rates.flatten(), rates.flatten())
        cost += self.rate_weight * rate_magnitude_sq * dt
        
        # Control effort (penalize large position commands away from goal)
        if control is not None:
            control = control.reshape(-1, 1)
            control_error = control - self.goal_position  # Want to command near goal
            cost += self.control_weight * np.dot(control_error.flatten(), control_error.flatten()) * dt
            
            # Control smoothness (penalize rapid changes)
            if self.prev_control is not None:
                control_change = control - self.prev_control
                cost += self.control_change_weight * np.dot(control_change.flatten(), control_change.flatten()) * dt
            
            self.prev_control = control.copy()
        
        # Collision costs (integrated over time)
        if 'collision_result' in kwargs:
            collision_result = kwargs['collision_result']
            
            if collision_result.has_collision:
                self.collision_count += 1
                if collision_result.is_out_of_bounds:
                    collision_cost = self.collision_weight * 2.0 * dt
                else:
                    collision_cost = self.collision_weight * (1.0 + collision_result.total_penetration_depth) * dt
                
                cost += collision_cost
            
            # Proximity gradient for smooth optimization (integrated over time)
            if not collision_result.has_collision and collision_result.min_obstacle_distance < float('inf'):
                safe_distance = 0.5
                if collision_result.min_obstacle_distance < safe_distance:
                    proximity_factor = (safe_distance - collision_result.min_obstacle_distance) / safe_distance
                    cost += self.proximity_weight * (proximity_factor ** 2) * dt
        
        # Terminal cost
        if is_terminal:
            terminal_pos_cost = self.terminal_pos_weight * np.dot(pos_error.flatten(), pos_error.flatten())
            cost += terminal_pos_cost
            
            distance = np.linalg.norm(pos_error)
            if cost < self.best_cost:
                self.best_cost = cost
                self.best_state = {'position': pos.flatten(), 'distance': distance}
        
        return cost


cost_func = QuadrotorPositionCost(
    goal_position,
    pos_weight=100.0,           # Position tracking
    terminal_pos_weight=5000.0, # Strong terminal cost
    euler_weight=5.0,          # Increased: penalize large angles more
    rate_weight=5.0,            # Increased: penalize aggressive rates
    control_weight=0.0,         # Small penalty for commands away from goal
    control_change_weight=15.0, # Increased: demand smoother keyframe transitions
    collision_weight=10000.0,
    proximity_weight=50.0
)

# ============================================================================
# Input Function
# ============================================================================


horizon = 5  # seconds
num_keyframes = 15  # More keyframes = finer resolution, smoother control
control_dim = 3 # Position commands [x, y, z]

# Per-dimension bounds for position commands [x, y, z]
# X: [0, 10], Y: [-2.5, 2.5], Z: [-5, 0] (NED frame)
pos_lower_bounds = np.tile([0.0, -2.5, -5.0], num_keyframes)
pos_upper_bounds = np.tile([10.0, 2.5, 0.0], num_keyframes)

# Use piecewise constant (zero-order hold) instead of spline
# This eliminates overshoot and oscillations between keyframes
# it was just easier to debug for now, spline still may end up being better
input_func = PiecewiseConstantInput(
    numKeyframes=num_keyframes,
    totalSteps=int(horizon / physics_dt),
    control_dim=control_dim,
    u_min=np.array([0.0, -2.5, -5.0]),   # Min position bounds [x, y, z]
    u_max=np.array([10.0, 2.5, 0.0])     # Max position bounds [x, y, z]
)
print(f"Horizon: {horizon:.1f}s, Keyframes: {num_keyframes}, Physics dt: {physics_dt}s ({1/physics_dt:.0f}Hz)")
print(f"Input method: Piecewise Constant")

# ============================================================================
# Optimizer
# ============================================================================

decision_dim = num_keyframes * control_dim
optimizer = CrossEntropyMethod(
    population_size= 600,
    elite_frac=0.15,
    num_best_retained=0,
    max_iterations=100,  # Full optimization
    epsilon=1e-3,
    alpha=0.1,
    initial_std=2.0,  # Position space in meters
    min_std=0.25,
    bounds=(pos_lower_bounds, pos_upper_bounds),
    verbose=True
)
print(f"CEM Optimizer: pop=400, elites=80, iters=100, alpha=0.1, min_std=0.25")
print(f"Per-dimension bounds: X[0,10], Y[-2.5,2.5], Z[-5,0]")
print(f"Decision vector dimension: {decision_dim} ({num_keyframes} keyframes x {control_dim} dims)")

# ============================================================================
# NBBMPC Controller construction
# ============================================================================

mpc = NBBMPC(
    vehicle=vehicle,
    costFunction=cost_func,
    optimizer=optimizer,
    inputFunction=input_func,
    control_dim=control_dim,
    physics_dt=physics_dt,
    numControlKeyframes=num_keyframes,
    dynamicHorizon=False,
    maxHorizon=horizon,
    cost_context={'vehicle': vehicle, 'config_space': config_space},
    verbose=False  # Set to True to see detailed NBBMPC debug prints
)

mpc._x0 = x0.copy()
mpc.cost_context = {'vehicle': vehicle, 'config_space': config_space}

# ============================================================================
# Optimize
# ============================================================================

print("\nRunning MPC optimization...")
print("="*80)

# Initial guess: linear interpolation from start to goal
initial_keyframes = np.linspace(INIT_POS, goal_position, num_keyframes)
initial_guess = initial_keyframes.flatten()

print(f"Initial guess (first keyframe): {initial_keyframes[0]}")
print(f"Initial guess (last keyframe): {initial_keyframes[-1]}")

try:
    best_decision = optimizer.optimize(initial_guess=initial_guess, controller=mpc)
    print("="*80)
    optimization_completed = True
except KeyboardInterrupt:
    print("\n\nOptimization interrupted by user (KeyboardInterrupt).")
    # Get current best from optimizer
    if hasattr(optimizer, 'best_solution'):
        best_decision = optimizer.best_solution
    else:
        # Fallback: use initial guess
        best_decision = initial_guess
    optimization_completed = False

if interrupt_requested or not optimization_completed:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\nGenerating interrupted snapshot at {timestamp}...")
    
    # Prepare wall obstacles for plotting
    wall_obstacles = [
        {'center': np.array([3.5, 0.0, -1.25]), 'size': np.array([0.5, 5.0, 2.5])},
        {'center': np.array([6.5, 0.0, -3.75]), 'size': np.array([0.5, 5.0, 2.5])},
    ]
    
    plot_current_best(best_decision, input_func, vehicle, quad_dynamics, x0,
                      horizon, physics_dt, INIT_POS, goal_position, dim,
                      wall_obstacles, timestamp_suffix=f"interrupted_{timestamp}")
    
    print("\n" + "="*80)
    print("Interrupted optimization snapshot complete.")
    print("="*80)
    sys.exit(0)

print("="*80)

# ============================================================================
# Rollout and Visualization
# ============================================================================

print("\nGenerating trajectory rollout...")

# Set up input function
input_func.updateStartAndEndTimes(0.0, horizon)
input_func.updateKeyFrameValues(best_decision)

# Rollout trajectory
trajectory_states = []
quad_dynamics.set_state(x0.copy())
vehicle.set_state(x0.copy())
trajectory_states.append(vehicle.state.flatten().copy())

t = 0.0
rollout_steps = int(horizon / physics_dt)
for step in range(rollout_steps):
    pos_cmd = input_func.calculateInput(t)
    state = vehicle.propagate(physics_dt, u=pos_cmd)
    trajectory_states.append(state.flatten().copy())
    t += physics_dt

trajectory_states = np.array(trajectory_states)
traj_positions = trajectory_states[:, 0:3]
traj_velocities = trajectory_states[:, 3:6]
traj_euler = trajectory_states[:, 6:9]
traj_rates = trajectory_states[:, 9:12]

final_pos = trajectory_states[-1, 0:3]
final_vel = trajectory_states[-1, 3:6]

# Diagnostic: Check for aggressive control
max_vel = np.max(np.linalg.norm(traj_velocities, axis=1))
max_euler = np.max(np.abs(traj_euler) * 180/np.pi)
max_rates = np.max(np.abs(traj_rates) * 180/np.pi)

print(f"Final position: {final_pos}")
print(f"Final velocity: {final_vel}")
print(f"Position error: {np.linalg.norm(final_pos - goal_position):.4f}m")
print(f"\nTrajectory Statistics:")
print(f"  Max velocity: {max_vel:.2f} m/s")
print(f"  Max Euler angle: {max_euler:.2f} deg")
print(f"  Max body rate: {max_rates:.2f} deg/s")
if max_vel > 3.0:
    print(f"  WARNING: High velocities detected - increase control_change_weight")
if max_rates > 200:
    print(f"  WARNING: Aggressive body rates - increase control_change_weight or num_keyframes")

# ============================================================================
# Visualization - 3D Views and Projections
# ============================================================================

# Prepare obstacle geometry for visualization
# Wall 1: X=3.5, blocks Z=[-2.5, 0], center at Z=-1.25, height=2.5
# Wall 2: X=6.5, blocks Z=[-5, -2.5], center at Z=-3.75, height=2.5
wall_obstacles = [
    {'center': np.array([3.5, 0.0, -1.25]), 'size': np.array([0.5, 5.0, 2.5])},
    {'center': np.array([6.5, 0.0, -3.75]), 'size': np.array([0.5, 5.0, 2.5])},
]

fig1 = plt.figure(figsize=(18, 12))
ax1 = fig1.add_subplot(221, projection='3d')  # 3D view 1
ax2 = fig1.add_subplot(222, projection='3d')  # 3D view 2
ax3 = fig1.add_subplot(223)  # Top-down (X-Y)
ax4 = fig1.add_subplot(224)  # Side view (X-Z)

x_bounds = [dim[0], dim[1]]
y_bounds = [dim[2], dim[3]]
z_bounds = [dim[4], dim[5]]

# Helper to draw obstacles in 3D
def draw_walls_3d(ax):
    for wall in wall_obstacles:
        center = wall['center']
        size = wall['size']
        half = size / 2.0
        corners = center[:, np.newaxis] + np.array([
            [-half[0], -half[1], -half[2]], [half[0], -half[1], -half[2]],
            [half[0], half[1], -half[2]], [-half[0], half[1], -half[2]],
            [-half[0], -half[1], half[2]], [half[0], -half[1], half[2]],
            [half[0], half[1], half[2]], [-half[0], half[1], half[2]]
        ]).T
        faces = [
            [corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]],
            [corners[:, 4], corners[:, 5], corners[:, 6], corners[:, 7]],
            [corners[:, 0], corners[:, 1], corners[:, 5], corners[:, 4]],
            [corners[:, 2], corners[:, 3], corners[:, 7], corners[:, 6]],
            [corners[:, 0], corners[:, 3], corners[:, 7], corners[:, 4]],
            [corners[:, 1], corners[:, 2], corners[:, 6], corners[:, 5]]
        ]
        ax.add_collection3d(Poly3DCollection(faces, alpha=0.3, facecolor='gray', edgecolor='black', linewidth=0.5))

# Plot 3D view 1
ax1.plot(traj_positions[:, 0], traj_positions[:, 1], traj_positions[:, 2], 
         'b-', linewidth=2, label='Trajectory', alpha=0.8)
draw_walls_3d(ax1)
ax1.scatter(*INIT_POS, color='green', s=200, marker='o', label='Start', edgecolors='darkgreen', linewidths=2, zorder=5)
ax1.scatter(*goal_position, color='red', s=200, marker='*', label='Goal', edgecolors='darkred', linewidths=2, zorder=5)
ax1.scatter(*final_pos, color='blue', s=150, marker='x', label='Final', linewidths=3, zorder=5)
ax1.set_xlabel('X [m]')
ax1.set_ylabel('Y [m]')
ax1.set_zlabel('Z [m]')
ax1.set_xlim(x_bounds)
ax1.set_ylim(y_bounds)
ax1.set_zlim(z_bounds)
ax1.set_title('3D View 1 (Angle 1)', fontweight='bold')
ax1.view_init(elev=20, azim=45)
ax1.legend(loc='upper left', fontsize=8)
ax1.grid(True, alpha=0.3)

# Plot 3D view 2
ax2.plot(traj_positions[:, 0], traj_positions[:, 1], traj_positions[:, 2], 
         'b-', linewidth=2, label='Trajectory', alpha=0.8)
draw_walls_3d(ax2)
ax2.scatter(*INIT_POS, color='green', s=200, marker='o', label='Start', edgecolors='darkgreen', linewidths=2, zorder=5)
ax2.scatter(*goal_position, color='red', s=200, marker='*', label='Goal', edgecolors='darkred', linewidths=2, zorder=5)
ax2.scatter(*final_pos, color='blue', s=150, marker='x', label='Final', linewidths=3, zorder=5)
ax2.set_xlabel('X [m]')
ax2.set_ylabel('Y [m]')
ax2.set_zlabel('Z [m]')
ax2.set_xlim(x_bounds)
ax2.set_ylim(y_bounds)
ax2.set_zlim(z_bounds)
ax2.set_title('3D View 2 (Angle 2)', fontweight='bold')
ax2.view_init(elev=10, azim=135)
ax2.legend(loc='upper left', fontsize=8)
ax2.grid(True, alpha=0.3)

# Top-down view (X-Y)
ax3.plot(traj_positions[:, 0], traj_positions[:, 1], 'b-', linewidth=2, label='Trajectory', alpha=0.8)
for wall in wall_obstacles:
    center, size = wall['center'], wall['size']
    rect = plt.Rectangle((center[0]-size[0]/2, center[1]-size[1]/2), size[0], size[1],
                         facecolor='gray', alpha=0.3, edgecolor='black', linewidth=1)
    ax3.add_patch(rect)
ax3.scatter(*INIT_POS[0:2], color='green', s=200, marker='o', label='Start', edgecolors='darkgreen', linewidths=2, zorder=5)
ax3.scatter(*goal_position[0:2], color='red', s=200, marker='*', label='Goal', edgecolors='darkred', linewidths=2, zorder=5)
ax3.scatter(*final_pos[0:2], color='blue', s=150, marker='x', label='Final', linewidths=3, zorder=5)
ax3.set_xlabel('X [m]', fontweight='bold')
ax3.set_ylabel('Y [m]', fontweight='bold')
ax3.set_title('Top-Down View (X-Y)', fontweight='bold')
ax3.set_xlim(x_bounds)
ax3.set_ylim(y_bounds)
ax3.set_aspect('equal')
ax3.legend(loc='upper left', fontsize=8)
ax3.grid(True, alpha=0.3)

# Side view (X-Z) - CORRECTED: use X and Z dimensions properly
ax4.plot(traj_positions[:, 0], traj_positions[:, 2], 'b-', linewidth=2, label='Trajectory', alpha=0.8)
for wall in wall_obstacles:
    center, size = wall['center'], wall['size']
    # X-Z projection: bottom-left corner at (x - width/2, z - height/2)
    # Width = size[0] (X dimension), Height = size[2] (Z dimension)
    bottom_left_x = center[0] - size[0]/2
    bottom_left_z = center[2] - size[2]/2
    rect = plt.Rectangle((bottom_left_x, bottom_left_z), size[0], size[2],
                         facecolor='gray', alpha=0.3, edgecolor='black', linewidth=1)
    ax4.add_patch(rect)
ax4.scatter(INIT_POS[0], INIT_POS[2], color='green', s=200, marker='o', label='Start', edgecolors='darkgreen', linewidths=2, zorder=5)
ax4.scatter(goal_position[0], goal_position[2], color='red', s=200, marker='*', label='Goal', edgecolors='darkred', linewidths=2, zorder=5)
ax4.scatter(final_pos[0], final_pos[2], color='blue', s=150, marker='x', label='Final', linewidths=3, zorder=5)
ax4.set_xlabel('X [m]', fontweight='bold')
ax4.set_ylabel('Z [m]', fontweight='bold')
ax4.set_title('Side View (X-Z)', fontweight='bold')
ax4.set_xlim(x_bounds)
ax4.set_ylim(z_bounds)
ax4.set_aspect('equal')
ax4.legend(loc='upper left', fontsize=8)
ax4.grid(True, alpha=0.3)

fig1.suptitle('Quadrotor MPC - 3D Views and Projections', fontsize=14, fontweight='bold', y=0.98)
output_path1 = Path(__file__).parent / "quadrotor_mpc_3d_views.png"
plt.tight_layout()
plt.savefig(output_path1, dpi=150, bbox_inches='tight')
print(f"\n3D views saved: {output_path1}")
plt.close()

# ============================================================================
# Visualization - Tracking Performance
# ============================================================================

fig2 = plt.figure(figsize=(18, 12))

# Position vs time with goal
ax1 = fig2.add_subplot(221)
time_vec = np.linspace(0, horizon, len(traj_positions))
ax1.plot(time_vec, traj_positions[:, 0], 'b-', linewidth=2, label='X', alpha=0.8)
ax1.plot(time_vec, traj_positions[:, 1], 'g-', linewidth=2, label='Y', alpha=0.8)
ax1.plot(time_vec, traj_positions[:, 2], 'r-', linewidth=2, label='Z', alpha=0.8)
ax1.axhline(goal_position[0], color='b', linestyle='--', alpha=0.4, linewidth=1.5, label='X goal')
ax1.axhline(goal_position[1], color='g', linestyle='--', alpha=0.4, linewidth=1.5, label='Y goal')
ax1.axhline(goal_position[2], color='r', linestyle='--', alpha=0.4, linewidth=1.5, label='Z goal')
ax1.set_xlabel('Time [s]', fontweight='bold')
ax1.set_ylabel('Position [m]', fontweight='bold')
ax1.set_title('Position vs Time (Goal Tracking)', fontweight='bold')
ax1.legend(loc='best', fontsize=8, ncol=2)
ax1.grid(True, alpha=0.3)

# Velocity
ax2 = fig2.add_subplot(222)
ax2.plot(time_vec, traj_velocities[:, 0], linewidth=2, label='Vx', alpha=0.8)
ax2.plot(time_vec, traj_velocities[:, 1], linewidth=2, label='Vy', alpha=0.8)
ax2.plot(time_vec, traj_velocities[:, 2], linewidth=2, label='Vz', alpha=0.8)
ax2.axhline(0, color='k', linestyle='--', alpha=0.3, linewidth=1)
ax2.set_xlabel('Time [s]', fontweight='bold')
ax2.set_ylabel('Velocity [m/s]', fontweight='bold')
ax2.set_title('Velocity vs Time', fontweight='bold')
ax2.legend(loc='best', fontsize=8)
ax2.grid(True, alpha=0.3)

# Euler angles (LQR regulated)
ax3 = fig2.add_subplot(223)
ax3.plot(time_vec, np.rad2deg(traj_euler[:, 0]), linewidth=2, label='Roll', alpha=0.8)
ax3.plot(time_vec, np.rad2deg(traj_euler[:, 1]), linewidth=2, label='Pitch', alpha=0.8)
ax3.plot(time_vec, np.rad2deg(traj_euler[:, 2]), linewidth=2, label='Yaw', alpha=0.8)
ax3.axhline(0, color='k', linestyle='--', alpha=0.3, linewidth=1)
ax3.set_xlabel('Time [s]', fontweight='bold')
ax3.set_ylabel('Angle [deg]', fontweight='bold')
ax3.set_title('Euler Angles (LQR Regulated)', fontweight='bold')
ax3.legend(loc='best', fontsize=8)
ax3.grid(True, alpha=0.3)

# Body rates (LQR regulated)
ax4 = fig2.add_subplot(224)
ax4.plot(time_vec, np.rad2deg(traj_rates[:, 0]), linewidth=2, label='p (roll rate)', alpha=0.8)
ax4.plot(time_vec, np.rad2deg(traj_rates[:, 1]), linewidth=2, label='q (pitch rate)', alpha=0.8)
ax4.plot(time_vec, np.rad2deg(traj_rates[:, 2]), linewidth=2, label='r (yaw rate)', alpha=0.8)
ax4.axhline(0, color='k', linestyle='--', alpha=0.3, linewidth=1)
ax4.set_xlabel('Time [s]', fontweight='bold')
ax4.set_ylabel('Rate [deg/s]', fontweight='bold')
ax4.set_title('Body Rates (LQR Regulated)', fontweight='bold')
ax4.legend(loc='best', fontsize=8)
ax4.grid(True, alpha=0.3)

fig2.suptitle('Quadrotor MPC - Tracking Performance', fontsize=14, fontweight='bold', y=0.98)
output_path2 = Path(__file__).parent / "quadrotor_mpc_tracking.png"
plt.tight_layout()
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"Tracking plots saved: {output_path2}")
plt.close()

# ============================================================================
# Evaluate Results
# ============================================================================

if cost_func.best_state:
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Best cost: {cost_func.best_cost:.2f}")
    print(f"Final position error: {cost_func.best_state['distance']:.4f}m")
    print(f"Final velocity magnitude: {np.linalg.norm(final_vel):.4f}m/s")
    
    # Success criteria
    pos_threshold = 0.5  # meters
    vel_threshold = 0.5  # m/s
    
    pos_ok = cost_func.best_state['distance'] < pos_threshold
    vel_ok = np.linalg.norm(final_vel) < vel_threshold
    
    print(f"\nPosition error < {pos_threshold}m: {'PASS' if pos_ok else ' FAIL'}")
    print(f"Velocity < {vel_threshold}m/s: {'PASS' if vel_ok else ' FAIL'}")
    
    if pos_ok and vel_ok:
        print("\nTEST PASSED - Quadrotor MPC with LQR inner loop validated!")
    else:
        print("\n TEST FAILED - Trajectory did not meet success criteria")
    print("="*80)
