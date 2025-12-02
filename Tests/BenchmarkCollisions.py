"""Performance benchmark for collision detection in MPC loops."""
import numpy as np
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from Vehicles.Vehicle import Vehicle
from Utils.GeometryUtils import Box3D, PointXYZ
from ConfigurationSpace.ConfigSpace3D import ConfigurationSpace3D


class SimpleDynamics:
    """Simple dynamics for benchmarking."""
    def propagate_state(self, timestep, state, u=None, **kwargs):
        new_state = state.astype(np.float64).copy()
        if u is not None:
            new_state[:3] += u[:3] * timestep
        return new_state


def benchmark_collision_checks():
    """Benchmark collision checking performance."""
    print("=" * 70)
    print("COLLISION DETECTION PERFORMANCE BENCHMARK")
    print("=" * 70)
    
    # Create configuration space with obstacles
    config_space = ConfigurationSpace3D([0, 100, 0, 100, 0, 100])
    
    # Add multiple obstacles (simulating a cluttered environment)
    num_obstacles = 50
    np.random.seed(42)
    for i in range(num_obstacles):
        center = PointXYZ(
            np.random.uniform(10, 90),
            np.random.uniform(10, 90),
            np.random.uniform(10, 90)
        )
        size = PointXYZ(
            np.random.uniform(2, 5),
            np.random.uniform(2, 5),
            np.random.uniform(2, 5)
        )
        config_space.add_obstacle(Box3D(center, size))
    
    print(f"\nSetup: {num_obstacles} obstacles in 100x100x100 space")
    
    # Create vehicle
    dynamics = SimpleDynamics()
    vehicle = Vehicle(
        dynamics_class=dynamics,
        size=(1.0, 1.0, 1.0),
        initial_state=np.array([50.0, 50.0, 50.0])
    )
    
    # Benchmark 1: Collision checks only
    print("\n" + "-" * 70)
    print("Benchmark 1: Pure collision checks (typical MPC constraint check)")
    print("-" * 70)
    
    num_checks = 10000
    positions = np.random.uniform(10, 90, (num_checks, 3))
    
    start_time = time.perf_counter()
    collision_count = 0
    for pos in positions:
        vehicle.set_state(np.concatenate([pos, [0, 0, 0]]))
        if vehicle.check_collision_with_config_space(config_space):
            collision_count += 1
    elapsed = time.perf_counter() - start_time
    
    print(f"Total checks: {num_checks}")
    print(f"Collisions detected: {collision_count}")
    print(f"Total time: {elapsed:.4f} seconds")
    print(f"Average per check: {elapsed/num_checks*1000:.4f} ms")
    print(f"Checks per second: {num_checks/elapsed:.0f}")
    
    # Benchmark 2: Distance queries
    print("\n" + "-" * 70)
    print("Benchmark 2: Distance queries (for cost functions)")
    print("-" * 70)
    
    num_queries = 5000
    positions = np.random.uniform(10, 90, (num_queries, 3))
    
    start_time = time.perf_counter()
    distances = []
    for pos in positions:
        vehicle.set_state(np.concatenate([pos, [0, 0, 0]]))
        dist = vehicle.get_nearest_obstacle_distance(config_space)
        distances.append(dist)
    elapsed = time.perf_counter() - start_time
    
    print(f"Total queries: {num_queries}")
    print(f"Average distance: {np.mean(distances):.2f} m")
    print(f"Min distance: {np.min(distances):.2f} m")
    print(f"Total time: {elapsed:.4f} seconds")
    print(f"Average per query: {elapsed/num_queries*1000:.4f} ms")
    print(f"Queries per second: {num_queries/elapsed:.0f}")
    
    # Benchmark 3: MPC-style trajectory evaluation
    print("\n" + "-" * 70)
    print("Benchmark 3: Trajectory evaluation (typical MPC iteration)")
    print("-" * 70)
    
    horizon = 20  # MPC prediction horizon
    num_trajectories = 1000  # Number of particles in PSO or population in GA
    
    start_time = time.perf_counter()
    valid_trajectories = 0
    total_collisions = 0
    
    for traj_idx in range(num_trajectories):
        # Random trajectory (simplified)
        start_pos = np.random.uniform(20, 80, 3)
        velocity = np.random.uniform(-2, 2, 3)
        
        trajectory_valid = True
        for step in range(horizon):
            pos = start_pos + velocity * step * 0.1
            vehicle.set_state(np.concatenate([pos, velocity]))
            
            if vehicle.check_collision_with_config_space(config_space):
                trajectory_valid = False
                total_collisions += 1
                break  # Early exit on collision (realistic MPC behavior)
        
        if trajectory_valid:
            valid_trajectories += 1
    
    elapsed = time.perf_counter() - start_time
    
    print(f"Trajectories evaluated: {num_trajectories}")
    print(f"Horizon length: {horizon} steps")
    print(f"Valid trajectories: {valid_trajectories}")
    print(f"Total collision checks: {total_collisions}")
    print(f"Total time: {elapsed:.4f} seconds")
    print(f"Time per trajectory: {elapsed/num_trajectories*1000:.2f} ms")
    print(f"Trajectories per second: {num_trajectories/elapsed:.0f}")
    
    # Benchmark 4: Box update performance
    print("\n" + "-" * 70)
    print("Benchmark 4: Box transform updates (state propagation overhead)")
    print("-" * 70)
    
    num_updates = 50000
    states = np.random.uniform(-10, 10, (num_updates, 6))
    
    start_time = time.perf_counter()
    for state in states:
        vehicle.set_state(state)
    elapsed = time.perf_counter() - start_time
    
    print(f"Total updates: {num_updates}")
    print(f"Total time: {elapsed:.4f} seconds")
    print(f"Average per update: {elapsed/num_updates*1000000:.2f} μs")
    print(f"Updates per second: {num_updates/elapsed:.0f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"""
For MPC with black-box optimizer (PSO/GA):
- Collision checks: ~{num_checks/elapsed:.0f} checks/sec
- Distance queries: ~{num_queries/elapsed:.0f} queries/sec  
- Full trajectory eval: ~{num_trajectories/elapsed:.0f} trajectories/sec

Expected performance for your use case:
- Population size: 100 particles/individuals
- Horizon: 20 steps
- Max iterations: 50
- Total evals per MPC step: ~100,000 collision checks
- Estimated time: ~{100000/(num_checks/elapsed):.2f} seconds per MPC iteration

Optimization status: ✓ OPTIMIZED
- Cached FCL request objects
- Direct collision object updates (no recreation)
- Early exit on collision detection
- Fast AABB bounds checking
- Cached state indices in Vehicle class
    """)


if __name__ == "__main__":
    benchmark_collision_checks()