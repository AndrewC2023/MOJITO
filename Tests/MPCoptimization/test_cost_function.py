"""Quick test of QuadSlitCostFunction to verify it works correctly."""

import numpy as np
import sys
from pathlib import Path

# Add project source to path
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir / "src"))

from ConfigurationSpace.ConfigSpace3D import CollisionQueryResult, ObstacleProximity

# Import the test file to get the cost function
sys.path.append(str(Path(__file__).parent))
from QuadRestrictionTest import QuadSlitCostFunction

# Test parameters
goal_state = np.array([8.5, 1.25, -3.5, 0, 0, 0, 0, 0, 0, 0, 0, 0])
goal_position = np.array([8.5, 1.25, -3.5])
Q = np.diag([10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.5, 0.5, 0.5])
R = np.diag([0.1, 0.1, 0.1, 0.1])

cost_fn = QuadSlitCostFunction(
    goal_state=goal_state,
    goal_position=goal_position,
    Q=Q,
    R=R,
    collision_weight=100.0,
    boundary_weight=10000.0,
    proximity_weight=2.0,
    proximity_threshold=0.4,
    goal_weight=10.0,
    terminal_goal_weight=5000.0
)

# Test case 1: No collision, far from goal, NOT terminal
print("Test 1: No collision, far from goal, NOT terminal")
state = np.array([1.5, -1.25, -1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0])
control = np.array([0.25, 0.25, 0.25, 0.25])
dt = 0.1

collision_result = CollisionQueryResult(
    has_collision=False,
    is_out_of_bounds=False,
    obstacle_data=[],
    total_penetration_depth=0.0,
    num_collisions=0,
    min_obstacle_distance=2.0
)

# Calculate distance for reference
pos_distance = np.linalg.norm(state[0:3] - goal_position)
print(f"  Distance to goal: {pos_distance:.2f}m")
cost = cost_fn.evaluate(state, control, dt, collision_result=collision_result, is_terminal=False)
print(f"  Cost (non-terminal): {cost:.2f}")
print(f"  Expected breakdown: Q-term + R-term + goal(10 × {pos_distance:.1f}² × 0.1) ≈ {10 * pos_distance**2 * 0.1:.1f}")

# Test case 1b: Same state but TERMINAL
print("\nTest 1b: Same state, but IS terminal")
cost_terminal = cost_fn.evaluate(state, control, dt, collision_result=collision_result, is_terminal=True)
print(f"  Cost (terminal): {cost_terminal:.2f}")
print(f"  Expected additional: terminal(5000 × {pos_distance:.1f}² × 0.1) ≈ {5000 * pos_distance**2 * 0.1:.1f}")

# Test case 2: Collision detected
print("\nTest 2: Collision detected")
collision_result_collision = CollisionQueryResult(
    has_collision=True,
    is_out_of_bounds=False,
    obstacle_data=[ObstacleProximity(0, -0.05, True, 0.05)],
    total_penetration_depth=0.05,
    num_collisions=1,
    min_obstacle_distance=0.0
)

cost = cost_fn.evaluate(state, control, dt, collision_result=collision_result_collision, is_terminal=False)
print(f"Cost (with collision): {cost:.2f}")

# Test case 3: Terminal cost at goal
print("\nTest 3: Terminal cost at goal")
state_at_goal = goal_state.copy()
collision_result_clear = CollisionQueryResult(
    has_collision=False,
    is_out_of_bounds=False,
    obstacle_data=[],
    total_penetration_depth=0.0,
    num_collisions=0,
    min_obstacle_distance=5.0
)

cost = cost_fn.evaluate(state_at_goal, control, dt, collision_result=collision_result_clear, is_terminal=True)
print(f"Cost (at goal, terminal): {cost:.2f}")

# Test case 4: Out of bounds
print("\nTest 4: Out of bounds")
collision_result_oob = CollisionQueryResult(
    has_collision=True,
    is_out_of_bounds=True,
    obstacle_data=[],
    total_penetration_depth=0.0,
    num_collisions=0,
    min_obstacle_distance=float('inf')
)

cost = cost_fn.evaluate(state, control, dt, collision_result=collision_result_oob, is_terminal=False)
print(f"Cost (out of bounds): {cost:.2f}")

# Test case 5: Close proximity to obstacle
print("\nTest 5: Close proximity to obstacle (0.3m away, threshold 0.4m)")
collision_result_prox = CollisionQueryResult(
    has_collision=False,
    is_out_of_bounds=False,
    obstacle_data=[ObstacleProximity(0, 0.3, False, 0.0)],
    total_penetration_depth=0.0,
    num_collisions=0,
    min_obstacle_distance=0.3  # Within threshold of 0.4
)

cost = cost_fn.evaluate(state, control, dt, collision_result=collision_result_prox, is_terminal=False)
log_penalty = 2.0 * (-np.log(0.3 / 0.4)) * 0.1
print(f"  Cost (with proximity): {cost:.2f}")
print(f"  Proximity penalty alone: {log_penalty:.2f}")
print(f"  Total = base cost + proximity penalty")

print("\n All tests completed successfully!")
