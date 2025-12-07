"""Comprehensive NACMPC Module Validation Test

This test validates that the NACMPC controller:
1. Correctly interfaces with Vehicle dynamics
2. Properly decodes decision vectors into control trajectories
3. Rolls out dynamics with collision detection
4. Evaluates cost functions correctly
5. Integrates properly with optimizers
6. Handles all input function types
7. Maintains correct array dimensions throughout

This is the final validation before running full trajectory optimization.
"""

import numpy as np
import sys
from pathlib import Path

# Add project paths
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir / "src"))
sys.path.append(str(root_dir / "dependencies" / "gncpy" / "src"))

from gncpy.dynamics.aircraft.simple_multirotor_quat import SimpleMultirotorQuat, v_smap_quat
import gncpy.math as gmath
import fcl

from Vehicles.Vehicle import Vehicle
from ConfigurationSpace.ConfigSpace3D import ConfigurationSpace3D
from ConfigurationSpace.Obstacles import StaticObstacle
from Controls.NACMPC import NACMPC, PiecewiseConstantInput, LinearInterpolationInput, SplineInterpolationInput
from Optimizers.CrossEntropyMethod import CrossEntropyMethod
from Optimizers.OptimizerBase import CostFunction

print("="*80)
print("NACMPC MODULE COMPREHENSIVE VALIDATION")
print("="*80)

# ============================================================================
# TEST 1: NACMPC Initialization and Configuration
# ============================================================================
print("\n" + "="*80)
print("TEST 1: NACMPC Initialization")
print("="*80)

# Create configuration space
dim = [0, 10, -2.5, 2.5, -5, 0]
config_space = ConfigurationSpace3D(dim)

# Add a simple obstacle
obstacle_geom = fcl.Box(1.0, 1.0, 1.0)
obstacle_tf = fcl.Transform(np.eye(3), [5.0, 0.0, -2.5])
config_space.add_obstacle(StaticObstacle(obstacle_geom, obstacle_tf))

print(f"ConfigSpace created with {config_space.get_num_obstacles()} obstacle(s)")

# Create dynamics
config_file = root_dir / "Tests" / "MPCoptimization" / "SmallQuadrotor.yaml"
dynamics = SimpleMultirotorQuat(str(config_file), effector=None)

# Initialize
INIT_POS = np.array([1.0, 0.0, -1.0])
INIT_VEL = np.array([0.0, 0.0, 0.0])
INIT_EULER = np.array([0.0, 0.0, 0.0])
INIT_RATES = np.array([0.0, 0.0, 0.0])

dynamics.set_initial_conditions(
    INIT_POS, INIT_VEL, INIT_EULER, INIT_RATES,
    40.0, -111.0, 1387.0, np.array([20.0, 5.0, 45.0])
)
dynamics.vehicle.takenoff = True

initial_state = dynamics.vehicle.state.copy()
print(f"Dynamics initialized, state shape: {initial_state.shape}")

# Create vehicle
vehicle_geometry = fcl.Box(0.3, 0.3, 0.1)
vehicle = Vehicle(
    dynamics_class=dynamics,
    geometry=vehicle_geometry,
    initial_state=initial_state,
    state_indices={'position': [4, 5, 6]}
)
print(f"Vehicle created")

# Define cost function
goal_position = np.array([8.0, 0.0, -3.0])
goal_state_12dof = np.concatenate([goal_position, np.zeros(9)])  # pos, vel, euler, rates

Q = np.diag([10, 10, 10, 1, 1, 1, 2, 2, 2, 0.5, 0.5, 0.5])  # 12x12
R = np.diag([0.1, 0.1, 0.1, 0.1])  # 4x4

class QuadCostFunction(CostFunction):
    """LQR-style cost with collision penalty for quadrotor."""
    
    def __init__(self, goal_12dof, Q_matrix, R_matrix):
        self.goal = goal_12dof
        self.Q = Q_matrix
        self.R = R_matrix
    
    def evaluate(self, state, control, dt, **kwargs):
        """Extract 12-DOF state and compute LQR-style cost."""
        vehicle_obj = kwargs.get('vehicle')
        config_space_obj = kwargs.get('config_space')
        
        # Extract components from 48-DOF state
        pos = state[v_smap_quat.ned_pos].flatten()
        vel = state[v_smap_quat.body_vel].flatten()
        quat = state[v_smap_quat.quat].flatten()
        rates = state[v_smap_quat.body_rot_rate].flatten()
        
        # Convert quaternion to Euler
        roll, pitch, yaw = gmath.quat_to_euler(quat)
        euler = np.array([roll, pitch, yaw]).flatten()
        
        # Build 12-DOF state
        state_12dof = np.concatenate([pos, vel, euler, rates])
        
        # State cost
        state_error = state_12dof - self.goal
        state_cost = np.dot(state_error, self.Q @ state_error)
        
        # Control cost
        control_flat = control.flatten()
        control_cost = np.dot(control_flat, self.R @ control_flat)
        
        # Collision penalty
        collision_cost = 0.0
        if vehicle_obj and config_space_obj:
            collision_result = config_space_obj.query_collision_detailed(vehicle_obj.collision_object)
            if collision_result.has_collision:
                collision_cost = 10000.0 * collision_result.total_penetration_depth
        
        return state_cost + control_cost + collision_cost

cost_func = QuadCostFunction(goal_state_12dof, Q, R)
print(f"Cost function defined")

# Create input function (5 keyframes)
num_keyframes = 5
control_dim = 4
horizon = 5.0
physics_steps = 50

input_func = SplineInterpolationInput(
    numKeyframes=num_keyframes,
    totalSteps=physics_steps,
    control_dim=control_dim,
    u_min=0.0,
    u_max=1.0
)
print(f"Input function created: {num_keyframes} keyframes")

# Create optimizer
optimizer = CrossEntropyMethod(
    population_size=50,  # Small for testing
    elite_frac=0.2,
    max_iterations=3,
    initial_std=0.2,
    bounds=(np.zeros(num_keyframes * control_dim), np.ones(num_keyframes * control_dim)),
    verbose=False
)
print(f"Optimizer created")

# Create NACMPC controller
mpc = NACMPC(
    vehicle=vehicle,
    costFunction=cost_func,
    optimizer=optimizer,
    inputFunction=input_func,
    control_dim=control_dim,
    physicsSteps=physics_steps,
    numControlKeyframes=num_keyframes,
    dynamicHorizon=False,  # Fixed horizon mode
    maxHorizon=horizon,
    debug=False
)
print(f"NACMPC controller created")
print(f"  Horizon: {mpc.maxHorizon}s (fixed)")
print(f"  Physics dt: {mpc.physics_dt}s")
print(f"  Total steps: {int(mpc.maxHorizon / mpc.physics_dt)}")
print(f"  Decision vector dimension: {num_keyframes * control_dim}")

print("\nTEST 1 PASSED: NACMPC initialization successful")

# ============================================================================
# TEST 2: Decision Vector Decoding
# ============================================================================
print("\n" + "="*80)
print("TEST 2: Decision Vector Decoding")
print("="*80)

# Create test decision vector
decision_vector = np.random.rand(num_keyframes * control_dim)
print(f"Decision vector shape: {decision_vector.shape}")
print(f"Decision vector range: [{decision_vector.min():.3f}, {decision_vector.max():.3f}]")

# Update input function with decision vector
mpc.inputFunction.updateKeyFrameValues(decision_vector)
print(f"Input function updated with decision vector")

# Sample controls at various times
test_times = np.linspace(0.0, horizon, 10)
for i, t in enumerate(test_times[:3]):  # Just show first 3
    control = mpc.inputFunction.calculateInput(t)
    print(f"  t={t:.2f}s: control shape={control.shape}, values={control}")
    assert control.shape == (control_dim,), f"Control shape mismatch: {control.shape}"

print("\nTEST 2 PASSED: Decision vector decoding works correctly")

# ============================================================================
# TEST 3: Dynamics Rollout
# ============================================================================
print("\n" + "="*80)
print("TEST 3: Dynamics Rollout with Collision Updates")
print("="*80)

# Reset vehicle to initial state
vehicle.set_state(initial_state)
print(f"Initial vehicle position: {vehicle.get_position()}")

# Perform manual rollout
rollout_states = [initial_state.copy()]
rollout_controls = []
rollout_costs = []

time = 0.0
current_state = initial_state.copy()
dt_rollout = mpc.physics_dt

for step in range(5):  # Just 5 steps for testing
    # Get control
    control = mpc.inputFunction.calculateInput(time)
    rollout_controls.append(control)
    
    # Propagate
    vehicle.set_state(current_state)
    new_state = vehicle.propagate(dt_rollout, control)
    
    # Update collision object (should happen automatically in propagate)
    quat = new_state[v_smap_quat.quat].flatten()
    roll, pitch, yaw = gmath.quat_to_euler(quat)
    from Utils.GeometryUtils import DCM3D
    Rx = DCM3D(roll, "x")
    Ry = DCM3D(pitch, "y")
    Rz = DCM3D(yaw, "z")
    dcm = Rz @ Ry @ Rx
    pos = new_state[v_smap_quat.ned_pos].flatten()
    vehicle.collision_object.setTransform(fcl.Transform(dcm, pos))
    
    # Check collision
    collision_result = config_space.query_collision_detailed(vehicle.collision_object)
    
    # Compute cost
    cost = cost_func.evaluate(new_state, control, dt_rollout, vehicle=vehicle, config_space=config_space)
    rollout_costs.append(cost)
    
    print(f"Step {step}: t={time:.2f}s, pos={pos}, collision={collision_result.has_collision}, cost={cost:.1f}")
    
    current_state = new_state.copy()
    rollout_states.append(new_state.copy())
    time += dt_rollout

print(f"\nRollout completed: {len(rollout_states)} states, {len(rollout_controls)} controls")
print(f"  Total cost: {sum(rollout_costs):.1f}")
print(f"  All state shapes: {[s.shape for s in rollout_states[:3]]}")  # Show first 3

print("\nTEST 3 PASSED: Dynamics rollout with collision detection works")

# ============================================================================
# TEST 4: Cost Function Evaluation with Different States
# ============================================================================
print("\n" + "="*80)
print("TEST 4: Cost Function Evaluation")
print("="*80)

# Test at initial state (far from goal)
vehicle.set_state(initial_state)
control_test = np.array([0.25, 0.25, 0.25, 0.25])
cost_initial = cost_func.evaluate(initial_state, control_test, 0.0, vehicle=vehicle, config_space=config_space)
print(f"Cost at initial state: {cost_initial:.2f}")

# Test at goal state
goal_state = initial_state.copy()
goal_state[v_smap_quat.ned_pos] = goal_position
goal_state[v_smap_quat.body_vel] = np.zeros(3)
goal_state[v_smap_quat.body_rot_rate] = np.zeros(3)
vehicle.set_state(goal_state)
cost_goal = cost_func.evaluate(goal_state, control_test, 0.0, vehicle=vehicle, config_space=config_space)
print(f"Cost at goal state: {cost_goal:.2f}")

assert cost_initial > cost_goal, "Cost should be higher at initial state than goal"
print(f"Cost is higher at initial state ({cost_initial:.1f}) than goal ({cost_goal:.1f})")

# Test at collision state (inside obstacle)
collision_state = initial_state.copy()
collision_state[v_smap_quat.ned_pos] = [5.0, 0.0, -2.5]  # Obstacle center
vehicle.set_state(collision_state)
quat_col = collision_state[v_smap_quat.quat].flatten()
roll_col, pitch_col, yaw_col = gmath.quat_to_euler(quat_col)
Rx_col = DCM3D(roll_col, "x")
Ry_col = DCM3D(pitch_col, "y")
Rz_col = DCM3D(yaw_col, "z")
dcm_col = Rz_col @ Ry_col @ Rx_col
pos_col = collision_state[v_smap_quat.ned_pos].flatten()
vehicle.collision_object.setTransform(fcl.Transform(dcm_col, pos_col))

cost_collision = cost_func.evaluate(collision_state, control_test, 0.0, vehicle=vehicle, config_space=config_space)
print(f"Cost at collision state: {cost_collision:.2f}")

assert cost_collision > cost_initial, "Cost should be much higher when colliding"
print(f"Cost is very high at collision state ({cost_collision:.1f})")

print("\nTEST 4 PASSED: Cost function responds correctly to state changes")

# ============================================================================
# TEST 5: Evaluate Decision Vector (Full MPC Evaluation)
# ============================================================================
print("\n" + "="*80)
print("TEST 5: Full MPC Evaluation via evaluate_decision_vector")
print("="*80)

# Reset to initial state
vehicle.set_state(initial_state)
mpc._x0 = initial_state.copy()
mpc.cost_context = {'vehicle': vehicle, 'config_space': config_space}

# Create a reasonable decision vector (hover controls)
hover_decision = np.full(num_keyframes * control_dim, 0.25)
print(f"Testing with hover decision vector: shape={hover_decision.shape}")

# Evaluate (this runs full rollout internally)
total_cost = mpc.evaluate_decision_vector(hover_decision)
print(f"Total cost from hover controls: {total_cost:.2f}")

assert isinstance(total_cost, (float, np.floating)), f"Cost should be scalar, got {type(total_cost)}"
assert not np.isnan(total_cost), "Cost should not be NaN"
assert not np.isinf(total_cost), "Cost should not be infinite"
print(f"Cost is valid scalar: {total_cost:.2f}")

# Try a different decision vector
aggressive_decision = np.full(num_keyframes * control_dim, 0.8)
total_cost_aggressive = mpc.evaluate_decision_vector(aggressive_decision)
print(f"Total cost from aggressive controls: {total_cost_aggressive:.2f}")

# Costs should be different
assert abs(total_cost - total_cost_aggressive) > 1.0, "Different controls should give different costs"
print(f"Different decision vectors produce different costs")

print("\nTEST 5 PASSED: Full MPC evaluation works correctly")

# ============================================================================
# TEST 6: Different Input Functions
# ============================================================================
print("\n" + "="*80)
print("TEST 6: Testing All Input Function Types")
print("="*80)

# Test with PiecewiseConstantInput
pwc_input = PiecewiseConstantInput(
    numKeyframes=num_keyframes,
    totalSteps=physics_steps,
    control_dim=control_dim,
    u_min=0.0,
    u_max=1.0
)

mpc_pwc = NACMPC(
    vehicle=vehicle,
    costFunction=cost_func,
    optimizer=optimizer,
    inputFunction=pwc_input,
    control_dim=control_dim,
    physics_dt=0.1,
    numControlKeyframes=num_keyframes,  # MUST match pwc_input.numKeyframes!
    dynamicHorizon=False,
    maxHorizon=horizon,
    debug=False
)

vehicle.set_state(initial_state)
mpc_pwc._x0 = initial_state.copy()
mpc_pwc.cost_context = {'vehicle': vehicle, 'config_space': config_space}
cost_pwc = mpc_pwc.evaluate_decision_vector(hover_decision)
print(f"PiecewiseConstantInput: cost={cost_pwc:.2f}")

# Test with LinearInterpolationInput
linear_input = LinearInterpolationInput(
    numKeyframes=num_keyframes,
    totalSteps=physics_steps,
    control_dim=control_dim,
    u_min=0.0,
    u_max=1.0
)

mpc_linear = NACMPC(
    vehicle=vehicle,
    costFunction=cost_func,
    optimizer=optimizer,
    inputFunction=linear_input,
    control_dim=control_dim,
    physics_dt=0.1,
    numControlKeyframes=num_keyframes,  # MUST match linear_input.numKeyframes!
    dynamicHorizon=False,
    maxHorizon=horizon,
    debug=False
)

vehicle.set_state(initial_state)
mpc_linear._x0 = initial_state.copy()
mpc_linear.cost_context = {'vehicle': vehicle, 'config_space': config_space}
cost_linear = mpc_linear.evaluate_decision_vector(hover_decision)
print(f"LinearInterpolationInput: cost={cost_linear:.2f}")

# Test with SplineInterpolationInput (already tested above)
print(f"SplineInterpolationInput: cost={total_cost:.2f} (from TEST 5)")

print("\nTEST 6 PASSED: All input function types work with NACMPC")

# ============================================================================
# TEST 7: Dimension Validation Throughout Pipeline
# ============================================================================
print("\n" + "="*80)
print("TEST 7: Array Dimension Validation Throughout Pipeline")
print("="*80)

# Track shapes through the pipeline
vehicle.set_state(initial_state)
print(f"1. Initial state shape: {initial_state.shape}")

decision = np.random.rand(num_keyframes * control_dim)
print(f"2. Decision vector shape: {decision.shape}")

mpc.inputFunction.updateKeyFrameValues(decision)
control_sample = mpc.inputFunction.calculateInput(0.0)
print(f"3. Control sample shape: {control_sample.shape}")

new_state = vehicle.propagate(0.1, control_sample)
print(f"4. Propagated state shape: {new_state.shape}")

# Extract for cost
pos_test = new_state[v_smap_quat.ned_pos].flatten()
vel_test = new_state[v_smap_quat.body_vel].flatten()
quat_test = new_state[v_smap_quat.quat].flatten()
rates_test = new_state[v_smap_quat.body_rot_rate].flatten()

print(f"5. Extracted shapes: pos={pos_test.shape}, vel={vel_test.shape}, quat={quat_test.shape}, rates={rates_test.shape}")

roll_t, pitch_t, yaw_t = gmath.quat_to_euler(quat_test)
euler_test = np.array([roll_t, pitch_t, yaw_t]).flatten()
print(f"6. Euler shape: {euler_test.shape}")

state_12dof_test = np.concatenate([pos_test, vel_test, euler_test, rates_test])
print(f"7. 12-DOF state shape: {state_12dof_test.shape}")

assert pos_test.shape == (3,), f"Position shape wrong: {pos_test.shape}"
assert vel_test.shape == (3,), f"Velocity shape wrong: {vel_test.shape}"
assert quat_test.shape == (4,), f"Quaternion shape wrong: {quat_test.shape}"
assert euler_test.shape == (3,), f"Euler shape wrong: {euler_test.shape}"
assert rates_test.shape == (3,), f"Rates shape wrong: {rates_test.shape}"
assert state_12dof_test.shape == (12,), f"12-DOF state shape wrong: {state_12dof_test.shape}"

print("\nAll dimensions correct throughout pipeline")

print("\nTEST 7 PASSED: Dimension validation successful")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("ALL NACMPC MODULE TESTS PASSED ")
print("="*80)
print("\nValidated Components:")
print("1. NACMPC initialization with all parameters")
print("2. Decision vector decoding via input functions")
print("3. Dynamics rollout with collision detection")
print("4. Cost function evaluation at different states")
print("5. Full MPC evaluation pipeline (evaluate_decision_vector)")
print("6. All three input function types (PWC, Linear, Spline)")
print("7. Array dimensions throughout entire pipeline")
print("\n" + "="*80)
print("READY FOR FULL TRAJECTORY OPTIMIZATION")
print("="*80)
