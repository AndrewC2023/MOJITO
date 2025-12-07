"""Simple test: Can the cascaded controller descend 1 meter?"""
import numpy as np
import sys
import scipy.linalg
sys.path.insert(0, '/workspaces/MOJITO/dependencies/gncpy/src')

from gncpy.dynamics.aircraft.simple_multirotor_quat import SimpleMultirotorQuat, v_smap_quat
import gncpy.math as gmath

# Constants
MASS = 0.8  # kg
HOVER_CMD = 0.7
DT = 0.01

# Initial conditions
INIT_POS = np.array([0.0, 0.0, -2.0])
INIT_VEL = np.array([0.0, 0.0, 0.0])
INIT_EULER = np.array([0.0, 0.0, 0.0])
INIT_RATES = np.array([0.0, 0.0, 0.0])

# Simple descent target: 1 meter down
TARGET_POS = np.array([0.0, 0.0, -3.0])

# WGS84
REF_LAT, REF_LON, TERRAIN_ALT = 0.0, 0.0, 0.0
ned_mag = np.array([0.2217, 0.0000, 0.3999])

# Create vehicle
vehicle = SimpleMultirotorQuat(use_euler_angles_only=True, use_quat_in_state=True)
vehicle.set_initial_conditions(
    INIT_POS, INIT_VEL, INIT_EULER, INIT_RATES,
    REF_LAT, REF_LON, TERRAIN_ALT, ned_mag
)
vehicle.takenoff = True

# Design LQR
quat_trim = vehicle.state[v_smap_quat.quat]
roll_trim, pitch_trim, yaw_trim = gmath.quat_to_euler(quat_trim)
trim_state = np.concatenate([
    vehicle.state[v_smap_quat.ned_pos],
    vehicle.state[v_smap_quat.vel],
    np.array([roll_trim, pitch_trim, yaw_trim]),
    vehicle.state[v_smap_quat.body_rot_rate]
])

# Design LQR
quat_trim = vehicle.state[v_smap_quat.quat]
roll_trim, pitch_trim, yaw_trim = gmath.quat_to_euler(quat_trim)
trim_state = np.concatenate([
    vehicle.state[v_smap_quat.ned_pos],
    vehicle.state[v_smap_quat.body_vel],
    np.array([roll_trim, pitch_trim, yaw_trim]),
    vehicle.state[v_smap_quat.body_rot_rate]
])

# Linearize dynamics
def state_deriv(x, u, t):
    old_state = vehicle.state.copy()
    roll, pitch, yaw = x[6:9]
    quat = gmath.euler_to_quat(roll, pitch, yaw)
    
    vehicle.state[v_smap_quat.ned_pos] = x[0:3]
    vehicle.state[v_smap_quat.body_vel] = x[3:6]
    vehicle.state[v_smap_quat.quat] = quat
    vehicle.state[v_smap_quat.body_rot_rate] = x[9:12]
    
    dcm_e2b = gmath.quat_to_dcm(quat)
    gravity_body = dcm_e2b @ np.array([0, 0, 9.81 * MASS])
    prop_force, prop_mom = vehicle._calc_prop_force_mom(u)
    force = gravity_body + prop_force
    
    ned_vel = dcm_e2b.T @ x[3:6]
    body_accel = force / MASS + np.cross(x[3:6], x[9:12])
    
    s_phi, c_phi = np.sin(roll), np.cos(roll)
    t_theta, c_theta = np.tan(pitch), np.cos(pitch)
    c_theta = max(abs(c_theta), 1e-6) * np.sign(c_theta) if c_theta != 0 else 1e-6
    eul_dot_mat = np.array([
        [1, s_phi * t_theta, c_phi * t_theta],
        [0, c_phi, -s_phi],
        [0, s_phi / c_theta, c_phi / c_theta],
    ])
    euler_rates = eul_dot_mat @ x[9:12]
    
    inertia = np.array(vehicle.params.mass.inertia_kgm2)
    body_rot_accel = np.linalg.inv(inertia) @ (
        prop_mom - np.cross(x[9:12], inertia @ x[9:12])
    )
    
    vehicle.state = old_state
    return np.concatenate([ned_vel, body_accel, euler_rates, body_rot_accel])

trim_input = np.ones(4) * HOVER_CMD
A, B = gmath.linearize_dynamics(state_deriv, trim_state, trim_input)

Q = scipy.linalg.block_diag(
    np.diag([50.0, 50.0, 50.0]),
    np.diag([0.1, 0.1, 0.1]),
    np.diag([5.0, 5.0, 5.0]),
    np.diag([0.5, 0.5, 0.5]),
)
R = np.diag([0.01, 0.01, 0.01, 0.01])

P = scipy.linalg.solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P

print(f"""
================================================================================
SIMPLE DESCENT TEST: Z=-2 Z=-3 (1 meter down)
================================================================================
""")

# Cascaded controller gains
Kp_pos = 1.5
Kd_vel = 2.0

# Simulate 2 seconds
N_STEPS = 200
for step in range(N_STEPS):
    state = vehicle.state
    pos = state[v_smap_quat.ned_pos]
    vel_body = state[v_smap_quat.body_vel]
    quat = state[v_smap_quat.quat]
    rates = state[v_smap_quat.body_rot_rate]
    euler = gmath.quat_to_euler(quat)
    
    # === OUTER LOOP: Position PD ===
    dcm_b2e = gmath.quat_to_dcm(quat).T
    vel_ned = dcm_b2e @ vel_body
    
    pos_error = TARGET_POS - pos
    vel_desired_ned = pos_error * Kp_pos
    vel_error = vel_desired_ned - vel_ned
    accel_desired_ned = vel_error * Kd_vel
    
    # Total acceleration (gravity compensation)
    accel_total_ned = accel_desired_ned + np.array([0.0, 0.0, 9.81])
    
    # Thrust magnitude
    thrust_needed_N = MASS * np.linalg.norm(accel_total_ned)
    thrust_cmd_base = np.sqrt(thrust_needed_N / (4 * 4.0))  # T = 4*cmd^2
    
    # Desired attitude from thrust direction
    thrust_dir = accel_total_ned / np.linalg.norm(accel_total_ned)
    desired_roll = np.arcsin(-thrust_dir[1])
    desired_pitch = np.arcsin(thrust_dir[0] / np.cos(desired_roll))
    desired_yaw = 0.0
    
    max_angle = np.deg2rad(20)
    desired_roll = np.clip(desired_roll, -max_angle, max_angle)
    desired_pitch = np.clip(desired_pitch, -max_angle, max_angle)
    
    # === INNER LOOP: LQR Attitude ===
    dcm_e2b = gmath.quat_to_dcm(quat)
    vel_desired_body = dcm_e2b @ vel_desired_ned
    
    x_current = np.array([
        pos[0], pos[1], pos[2],
        vel_body[0], vel_body[1], vel_body[2],
        euler[0], euler[1], euler[2],
        rates[0], rates[1], rates[2]
    ])
    
    # Target: CURRENT position (outer loop handles position)
    x_target = np.array([
        pos[0], pos[1], pos[2],  # CURRENT pos
        vel_desired_body[0], vel_desired_body[1], vel_desired_body[2],
        desired_roll, desired_pitch, desired_yaw,
        0.0, 0.0, 0.0
    ])
    
    state_error = x_current - x_target
    delta_u = -K @ state_error
    
    # Motor commands
    base_cmd = np.ones(4) * thrust_cmd_base
    motor_cmd = base_cmd + delta_u
    motor_cmd = np.clip(motor_cmd, 0.1, 0.95)
    
    if step == 0:
        print(f"Initial state:")
        print(f"  pos_error: {pos_error}")
        print(f"  accel_total_ned: {accel_total_ned}")
        print(f"  thrust_needed: {thrust_needed_N:.2f}N (hover: {MASS*9.81:.2f}N)")
        print(f"  thrust_cmd_base: {thrust_cmd_base:.3f} (hover: {HOVER_CMD:.3f})")
        print(f"  desired_pitch: {np.rad2deg(desired_pitch):.2f}°")
        print(f"  LQR pos_error: {state_error[0:3]}")
        print(f"  LQR vel_error: {state_error[3:6]}")
        print(f"  LQR euler_error: {np.rad2deg(state_error[6:9])}")
        print(f"  delta_u: {delta_u}")
        print(f"  motor_cmd: {motor_cmd}")
        print()
    
    # Propagate
    vehicle.propagate_state(DT, state, u=motor_cmd)
    
    if step < 10 or step % 20 == 0:
        print(f"t={step*DT:.2f}s: pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}], motors={motor_cmd}")

final_pos = vehicle.state[v_smap_quat.ned_pos]
print(f"""
================================================================================
Results:
  Start Z: {INIT_POS[2]:.3f}
  Final Z: {final_pos[2]:.3f}
  Target Z: {TARGET_POS[2]:.3f}
  
  {'DESCENDING!' if final_pos[2] < INIT_POS[2] else 'WENT UP!'}
================================================================================
""")
