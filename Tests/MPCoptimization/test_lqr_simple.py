"""Simple LQR test - check if quadrotor goes down when commanded down."""

import numpy as np
import scipy.linalg
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt

# Add paths
test_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(test_dir / ".." / ".." / "dependencies" / "gncpy" / "src"))

from gncpy.dynamics.aircraft.simple_multirotor_quat import (
    SimpleMultirotorQuat,
    v_smap_quat,
)
import gncpy.math as gmath


class VehicleWrapper:
    """Minimal Vehicle-like wrapper for SimpleMultirotorQuat matching MPC interface."""
    
    def __init__(self, quad: SimpleMultirotorQuat):
        self.quad = quad
    
    @property
    def state(self):
        """Return 12D Euler state: [ned_pos(3), body_vel(3), euler(3), body_rates(3)]."""
        quat = self.quad.vehicle.state[v_smap_quat.quat]
        roll, pitch, yaw = gmath.quat_to_euler(quat)
        
        return np.concatenate([
            self.quad.vehicle.state[v_smap_quat.ned_pos],
            self.quad.vehicle.state[v_smap_quat.body_vel],
            np.array([roll, pitch, yaw]),
            self.quad.vehicle.state[v_smap_quat.body_rot_rate],
        ])
    
    def propagate(self, dt, motor_cmd):
        """Propagate dynamics forward by dt with motor commands."""
        self.quad.propagate_state(dt, self.quad.vehicle.state, u=motor_cmd)


print("="*80)
print("SIMPLE LQR TEST - Does quadrotor go DOWN when commanded DOWN?")
print("="*80)

# Initialize vehicle
config_file = test_dir / ".." / ".." / "dependencies" / "gncpy" / "test" / "validation" / "simple_multirotor" / "small_quad_config.yaml"
quad = SimpleMultirotorQuat(str(config_file))
vehicle = VehicleWrapper(quad)

# Initial conditions (hover at z=-2.0)
INIT_POS = np.array([0.0, 0.0, -2.0])
INIT_VEL = np.array([0.0, 0.0, 0.0])
INIT_EULER = np.array([0.0, 0.0, 0.0])  # degrees
INIT_RATES = np.array([0.0, 0.0, 0.0])

REF_LAT, REF_LON, TERRAIN_ALT = 34.0, -86.0, 0.0
ned_mag = np.array([20.0, 5.0, 45.0])

vehicle.quad.set_initial_conditions(
    INIT_POS, INIT_VEL, INIT_EULER, INIT_RATES, 
    REF_LAT, REF_LON, TERRAIN_ALT, ned_mag
)
vehicle.quad.vehicle.takenoff = True

# Calculate hover command
mass = vehicle.quad.vehicle.params.mass.mass_kg
gravity = 9.81
hover_thrust_per_motor = mass * gravity / vehicle.quad.vehicle.params.motor.num_motors

thrust_poly = np.polynomial.Polynomial(vehicle.quad.vehicle.params.prop.poly_thrust[-1::-1])
roots = (thrust_poly - hover_thrust_per_motor).roots()
valid_roots = roots[np.isreal(roots) & (roots.real > 0) & (roots.real < 1)]
hover_cmd = float(valid_roots[0].real) if len(valid_roots) > 0 else 0.7

trim_motor_cmd = np.ones(vehicle.quad.vehicle.params.motor.num_motors) * hover_cmd
print(f"\nVehicle: mass={mass:.3f}kg, hover_cmd={hover_cmd:.3f}")
print(f"Initial position: {INIT_POS}")

# Linearize and design LQR (same as gncpy test)
def state_deriv(x, u, t):
    """12-DOF dynamics: [ned_pos(3), body_vel(3), euler(3), body_rates(3)]."""
    old_state = vehicle.quad.vehicle.state.copy()
    
    roll, pitch, yaw = x[6:9]
    quat = gmath.euler_to_quat(roll, pitch, yaw)
    
    vehicle.quad.vehicle.state[v_smap_quat.ned_pos] = x[0:3]
    vehicle.quad.vehicle.state[v_smap_quat.body_vel] = x[3:6]
    vehicle.quad.vehicle.state[v_smap_quat.quat] = quat
    vehicle.quad.vehicle.state[v_smap_quat.body_rot_rate] = x[9:12]
    
    dcm_e2b = gmath.quat_to_dcm(quat)
    gravity_body = dcm_e2b @ np.array([0, 0, gravity * mass])
    prop_force, prop_mom = vehicle.quad.vehicle._calc_prop_force_mom(u)
    force = gravity_body + prop_force
    
    ned_vel = dcm_e2b.T @ x[3:6]
    body_accel = force / mass + np.cross(x[3:6], x[9:12])
    
    s_phi, c_phi = np.sin(roll), np.cos(roll)
    t_theta, c_theta = np.tan(pitch), np.cos(pitch)
    c_theta = max(abs(c_theta), 1e-6) * np.sign(c_theta) if c_theta != 0 else 1e-6
    eul_dot_mat = np.array([
        [1, s_phi * t_theta, c_phi * t_theta],
        [0, c_phi, -s_phi],
        [0, s_phi / c_theta, c_phi / c_theta],
    ])
    euler_rates = eul_dot_mat @ x[9:12]
    
    inertia = np.array(vehicle.quad.vehicle.params.mass.inertia_kgm2)
    body_rot_accel = np.linalg.inv(inertia) @ (
        prop_mom - np.cross(x[9:12], inertia @ x[9:12])
    )
    
    vehicle.quad.vehicle.state = old_state
    return np.concatenate([ned_vel, body_accel, euler_rates, body_rot_accel])

# Get trim state
quat_trim = vehicle.quad.vehicle.state[v_smap_quat.quat]
roll_trim, pitch_trim, yaw_trim = gmath.quat_to_euler(quat_trim)

trim_state = np.concatenate([
    vehicle.quad.vehicle.state[v_smap_quat.ned_pos],
    vehicle.quad.vehicle.state[v_smap_quat.body_vel],
    np.array([roll_trim, pitch_trim, yaw_trim]),
    vehicle.quad.vehicle.state[v_smap_quat.body_rot_rate],
])

# Linearize
A, B = gmath.linearize_dynamics(state_deriv, trim_state, trim_motor_cmd)

# Design LQR with EXACT gncpy weights
Q = scipy.linalg.block_diag(
    np.diag([1.0, 1.0, 1.0]),     # position
    np.diag([0.1, 0.1, 0.1]),     # velocity
    np.diag([0.1, 0.1, 0.1]),     # euler angles - MATCH GNCPY
    np.diag([0.1, 0.1, 0.1]),     # rates - MATCH GNCPY
)
R = np.diag([1.0, 1.0, 1.0, 1.0])

P = scipy.linalg.solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P
print(f"\nLQR designed: K shape {K.shape}, max gain {np.max(np.abs(K)):.3f}")

# Test 1: Command X=2.0, Y=1.0, Z=-2.5 (EXACTLY like working gncpy test)
print("\n" + "="*80)
print("TEST: Command X=2.0, Y=1.0, Z=-3.0 (DESCEND 1m)")
print("="*80)

vehicle.quad.set_initial_conditions(
    INIT_POS, INIT_VEL, INIT_EULER, INIT_RATES,
    REF_LAT, REF_LON, TERRAIN_ALT, ned_mag
)
vehicle.quad.vehicle.takenoff = True

TARGET_POS = np.array([2.0, 1.0, -3.0])  # DESCEND 1m from hover
x_target = np.concatenate([TARGET_POS, np.zeros(9)])

DT = 0.01
SIM_TIME = 10.0  # MATCH GNCPY: 10 seconds
pos_history = []
euler_history = []
z_history = []
vz_history = []

for step in range(int(SIM_TIME / DT)):  # 1000 steps = 10 seconds like gncpy
    # Get current state using Vehicle wrapper
    x_current = vehicle.state
    
    # HOVER for first 0.1s like working test, then engage LQR
    if step < 10:
        motor_cmd = trim_motor_cmd.copy()
    else:
        delta_u = -K @ (x_current - x_target)
        motor_cmd = np.clip(trim_motor_cmd + delta_u, 0.1, 0.95)
    
    # Propagate using Vehicle wrapper
    vehicle.propagate(DT, motor_cmd)
    
    pos = vehicle.quad.vehicle.state[v_smap_quat.ned_pos]
    z_pos = pos[2]
    z_vel = vehicle.quad.vehicle.state[v_smap_quat.body_vel][2]
    
    # Get Euler angles
    quat = vehicle.quad.vehicle.state[v_smap_quat.quat]
    roll, pitch, yaw = gmath.quat_to_euler(quat)
    
    pos_history.append(pos.copy())
    euler_history.append([roll, pitch, yaw])
    z_history.append(z_pos)
    vz_history.append(z_vel)
    
    if step < 10 or step % 100 == 0:  # Print every second after hover
        print(f"  t={step*DT:.3f}s: pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}], motors={motor_cmd}")

pos_history = np.array(pos_history)
euler_history = np.array(euler_history)

final_pos = pos_history[-1]
print(f"\nFinal: pos=[{final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f}]")
print(f"Target: pos=[{TARGET_POS[0]:.3f}, {TARGET_POS[1]:.3f}, {TARGET_POS[2]:.3f}]")
print(f"Error: X={abs(final_pos[0] - TARGET_POS[0]):.3f}m, Y={abs(final_pos[1] - TARGET_POS[1]):.3f}m, Z={abs(final_pos[2] - TARGET_POS[2]):.3f}m")

# Check if it went DOWN initially
if z_history[5] > INIT_POS[2]:
    print("\nBUG: Quadrotor went UP when commanded DOWN!")
    print(f"   Start z={INIT_POS[2]:.4f}, after 0.05s z={z_history[5]:.4f}")
else:
    print("\nQuadrotor went DOWN as commanded")
    print(f"   Start z={INIT_POS[2]:.4f}, after 0.05s z={z_history[5]:.4f}")

print("\n" + "="*80)

# Plot Euler angles
time_vec = np.arange(len(euler_history)) * DT

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Position
axes[0, 0].plot(time_vec, pos_history[:, 0], label='X', linewidth=2)
axes[0, 0].plot(time_vec, pos_history[:, 1], label='Y', linewidth=2)
axes[0, 0].plot(time_vec, pos_history[:, 2], label='Z', linewidth=2)
axes[0, 0].axhline(TARGET_POS[0], color='blue', linestyle='--', alpha=0.3, label='X target')
axes[0, 0].axhline(TARGET_POS[1], color='orange', linestyle='--', alpha=0.3, label='Y target')
axes[0, 0].axhline(TARGET_POS[2], color='green', linestyle='--', alpha=0.3, label='Z target')
axes[0, 0].set_xlabel('Time [s]')
axes[0, 0].set_ylabel('Position [m]')
axes[0, 0].set_title('Position vs Time')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Euler angles
axes[0, 1].plot(time_vec, np.rad2deg(euler_history[:, 0]), label='Roll', linewidth=2)
axes[0, 1].plot(time_vec, np.rad2deg(euler_history[:, 1]), label='Pitch', linewidth=2)
axes[0, 1].plot(time_vec, np.rad2deg(euler_history[:, 2]), label='Yaw', linewidth=2)
axes[0, 1].set_xlabel('Time [s]')
axes[0, 1].set_ylabel('Angle [deg]')
axes[0, 1].set_title('Euler Angles (LQR Regulated)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axhline(0, color='gray', linestyle='-', alpha=0.2)

# Roll vs Yaw scatter
axes[1, 0].scatter(np.rad2deg(euler_history[:, 0]), np.rad2deg(euler_history[:, 2]), 
                   c=time_vec, cmap='viridis', s=10, alpha=0.6)
axes[1, 0].set_xlabel('Roll [deg]')
axes[1, 0].set_ylabel('Yaw [deg]')
axes[1, 0].set_title(f'Roll vs Yaw (correlation={np.corrcoef(euler_history[:,0], euler_history[:,2])[0,1]:.3f})')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axhline(0, color='gray', linestyle='-', alpha=0.2)
axes[1, 0].axvline(0, color='gray', linestyle='-', alpha=0.2)

# Statistics
stats_text = f"Euler Angle Statistics:\n"
stats_text += f"Roll:  [{np.min(euler_history[:,0])*180/np.pi:.1f}, {np.max(euler_history[:,0])*180/np.pi:.1f}] deg\n"
stats_text += f"Pitch: [{np.min(euler_history[:,1])*180/np.pi:.1f}, {np.max(euler_history[:,1])*180/np.pi:.1f}] deg\n"
stats_text += f"Yaw:   [{np.min(euler_history[:,2])*180/np.pi:.1f}, {np.max(euler_history[:,2])*180/np.pi:.1f}] deg\n"
stats_text += f"\nCorrelations:\n"
stats_text += f"Roll-Pitch: {np.corrcoef(euler_history[:,0], euler_history[:,1])[0,1]:.3f}\n"
stats_text += f"Roll-Yaw:   {np.corrcoef(euler_history[:,0], euler_history[:,2])[0,1]:.3f}\n"
stats_text += f"Pitch-Yaw:  {np.corrcoef(euler_history[:,1], euler_history[:,2])[0,1]:.3f}"

axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes,
                fontsize=11, verticalalignment='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
axes[1, 1].axis('off')

plt.tight_layout()
output_path = test_dir / "lqr_test_euler_angles.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nPlot saved: {output_path}")
print(stats_text)
