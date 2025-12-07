"""Collision Validation Test for Vehicle and ConfigSpace3D.

This test validates that:
1. Vehicle collision object is correctly positioned based on state
2. ConfigSpace3D correctly detects collisions and non-collisions
3. Rotation handling works correctly
4. State indexing from SimpleMultirotorQuat is properly handled
"""
import sys
import numpy as np
import fcl
from pathlib import Path

# Add project paths
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir / "src"))
sys.path.append(str(root_dir / "dependencies" / "gncpy" / "src"))

from gncpy.dynamics.aircraft.simple_multirotor_quat import SimpleMultirotorQuat, v_smap_quat
import gncpy.math as gmath
from ConfigurationSpace.ConfigSpace3D import ConfigurationSpace3D  # type: ignore
from ConfigurationSpace.Obstacles import StaticObstacle  # type: ignore
from Vehicles.Vehicle import Vehicle  # type: ignore
from Utils.GeometryUtils import DCM3D  # type: ignore


def print_box_corners(label: str, box_geom: fcl.Box, transform: fcl.Transform):
    """Print the corners of an FCL box for debugging."""
    # Get box half-extents
    x_half = box_geom.side[0] / 2
    y_half = box_geom.side[1] / 2
    z_half = box_geom.side[2] / 2
    
    # Define corners in local frame
    local_corners = np.array([
        [-x_half, -y_half, -z_half],
        [+x_half, -y_half, -z_half],
        [-x_half, +y_half, -z_half],
        [+x_half, +y_half, -z_half],
        [-x_half, -y_half, +z_half],
        [+x_half, -y_half, +z_half],
        [-x_half, +y_half, +z_half],
        [+x_half, +y_half, +z_half],
    ])
    
    # Transform to world frame
    rotation = transform.getRotation()
    translation = transform.getTranslation()
    world_corners = (rotation @ local_corners.T).T + translation
    
    print(f"\n{label}:")
    print(f"  Center: {translation}")
    print(f"  Size (x, y, z): ({box_geom.side[0]:.3f}, {box_geom.side[1]:.3f}, {box_geom.side[2]:.3f})")
    print(f"  Corners (world frame):")
    for i, corner in enumerate(world_corners):
        print(f"    {i}: [{corner[0]:7.3f}, {corner[1]:7.3f}, {corner[2]:7.3f}]")


def test_collision_at_state(config_file: Path):
    """Test collision detection at various vehicle states."""
    
    print("="*80)
    print("COLLISION VALIDATION TEST")
    print("="*80)
    
    # Create configuration space (simple 10x5x5 box)
    # NED frame: X=North, Y=East, Z=Down (negative up)
    # So zMin should be more negative (lower altitude), zMax less negative (higher altitude)
    dim = [0, 10, -2.5, 2.5, -5, 0]  # xMin, xMax, yMin, yMax, zMin, zMax (NED)
    config_space = ConfigurationSpace3D(dim)
    
    # Add a single obstacle: 1x1x1 box at center of space
    obstacle_geom = fcl.Box(1.0, 1.0, 1.0)
    obstacle_center = np.array([5.0, 0.0, -2.5])  # Center: x=5, y=0, z=-2.5 (midpoint between -5 and 0)
    obstacle_tf = fcl.Transform(np.eye(3), obstacle_center)
    config_space.add_obstacle(StaticObstacle(obstacle_geom, obstacle_tf))
    
    print("\n" + "-"*80)
    print("OBSTACLE CONFIGURATION")
    print("-"*80)
    print_box_corners("Obstacle Box", obstacle_geom, obstacle_tf)
    
    # Initialize dynamics
    QuadDynamics = SimpleMultirotorQuat(str(config_file), effector=None)
    
    # Vehicle geometry (0.3x0.3x0.1)
    vehicle_geometry = fcl.Box(0.3, 0.3, 0.1)
    
    # Test cases: [position, euler_angles, should_collide, description]
    test_cases = [
        # Case 1: Well clear of obstacle (initial position from main test)
        {
            'position': np.array([1.5, -1.25, -1.5]),
            'euler': np.array([0.0, 0.0, 0.0]),
            'should_collide': False,
            'description': 'Initial position - well clear'
        },
        # Case 2: Directly at obstacle center (definite collision)
        {
            'position': np.array([5.0, 0.0, -2.5]),
            'euler': np.array([0.0, 0.0, 0.0]),
            'should_collide': True,
            'description': 'At obstacle center'
        },
        # Case 3: Just touching obstacle edge
        {
            'position': np.array([4.35, 0.0, -2.5]),  # obstacle edge at 4.5, vehicle half-width 0.15
            'euler': np.array([0.0, 0.0, 0.0]),
            'should_collide': True,
            'description': 'Just touching obstacle edge'
        },
        # Case 4: Just clear of obstacle
        {
            'position': np.array([4.0, 0.0, -2.5]),  # obstacle edge at 4.5, vehicle half-width 0.15
            'euler': np.array([0.0, 0.0, 0.0]),
            'should_collide': False,
            'description': 'Just clear of obstacle'
        },
        # Case 5: Rotated 45 degrees about Z (yaw)
        {
            'position': np.array([1.5, -1.25, -1.5]),
            'euler': np.array([0.0, 0.0, 45.0]),
            'should_collide': False,
            'description': 'Initial position rotated 45 deg yaw'
        },
        # Case 6: Rotated 90 degrees about X (roll)
        {
            'position': np.array([1.5, -1.25, -1.5]),
            'euler': np.array([90.0, 0.0, 0.0]),
            'should_collide': False,
            'description': 'Initial position rotated 90 deg roll'
        },
    ]
    
    # Run test cases
    for i, test_case in enumerate(test_cases):
        print("\n" + "="*80)
        print(f"TEST CASE {i+1}: {test_case['description']}")
        print("="*80)
        
        position = test_case['position']
        euler_deg = test_case['euler']
        should_collide = test_case['should_collide']
        
        # Set vehicle initial conditions
        INIT_VEL = np.array([0.0, 0.0, 0.0])
        INIT_RATES = np.array([0.0, 0.0, 0.0])
        REF_LAT, REF_LON, TERRAIN_ALT = 40.0, -111.0, 1387.0
        ned_mag = np.array([20.0, 5.0, 45.0])
        
        QuadDynamics.set_initial_conditions(
            position, INIT_VEL, euler_deg, INIT_RATES, 
            REF_LAT, REF_LON, TERRAIN_ALT, ned_mag
        )
        QuadDynamics.vehicle.takenoff = True
        
        # Get the full state
        full_state = QuadDynamics.vehicle.state.copy()
        
        print(f"\nState vector information:")
        print(f"  Full state length: {len(full_state)}")
        print(f"  NED position (indices {v_smap_quat.ned_pos.value[0]}): {full_state[v_smap_quat.ned_pos]}")
        print(f"  Quaternion (indices {v_smap_quat.quat.value[0]}): {full_state[v_smap_quat.quat]}")
        print(f"  Body velocity (indices {v_smap_quat.body_vel.value[0]}): {full_state[v_smap_quat.body_vel]}")
        print(f"  Body rates (indices {v_smap_quat.body_rot_rate.value[0]}): {full_state[v_smap_quat.body_rot_rate]}")
        
        # Convert quaternion to Euler angles for DCM construction
        quat = full_state[v_smap_quat.quat]
        roll, pitch, yaw = gmath.quat_to_euler(quat)
        print(f"  Euler angles (deg): roll={np.rad2deg(roll):.2f}, pitch={np.rad2deg(pitch):.2f}, yaw={np.rad2deg(yaw):.2f}")
        
        # Build DCM using GeometryUtils (ZYX convention: rotation = Rz @ Ry @ Rx)
        Rx = DCM3D(roll, "x")
        Ry = DCM3D(pitch, "y")
        Rz = DCM3D(yaw, "z")
        dcm = Rz @ Ry @ Rx
        
        # Flatten DCM to row-major for state vector (if we were storing it)
        # For this test, we'll just pass position indices and update manually
        
        # Create vehicle with position indices only
        vehicle = Vehicle(
            dynamics_class=QuadDynamics,
            geometry=vehicle_geometry,
            initial_state=full_state,
            state_indices={
                'position': [4, 5, 6]  # v_smap_quat.ned_pos
            }
        )
        
        # Manually update collision object with rotation
        position = full_state[v_smap_quat.ned_pos]
        vehicle.collision_object.setTransform(fcl.Transform(dcm, position))
        
        # Print vehicle bounding box
        vehicle_tf = vehicle.get_transform()
        print_box_corners("Vehicle Box", vehicle_geometry, vehicle_tf)
        
        # Check collision
        collision_result = config_space.query_collision_detailed(vehicle.collision_object)
        
        print(f"\nCollision query result:")
        print(f"  has_collision: {collision_result.has_collision}")
        print(f"  is_out_of_bounds: {collision_result.is_out_of_bounds}")
        print(f"  num_collisions: {collision_result.num_collisions}")
        print(f"  total_penetration_depth: {collision_result.total_penetration_depth:.6f}")
        print(f"  min_obstacle_distance: {collision_result.min_obstacle_distance:.6f}")
        
        # Validate result
        if collision_result.has_collision == should_collide:
            print(f"\nTEST PASSED: Collision detection {'CORRECT' if should_collide else 'correctly no collision'}")
        else:
            print(f"\n TEST FAILED: Expected collision={should_collide}, got collision={collision_result.has_collision}")
        
        # Calculate manual distance check
        vehicle_center = full_state[v_smap_quat.ned_pos]
        manual_distance = np.linalg.norm(vehicle_center - obstacle_center)
        print(f"\nManual distance check:")
        print(f"  Vehicle center: {vehicle_center}")
        print(f"  Obstacle center: {obstacle_center}")
        print(f"  Euclidean distance: {manual_distance:.6f}")


if __name__ == "__main__":
    config_file = Path(__file__).parent / "MPCoptimization" / "SmallQuadrotor.yaml"
    
    if not config_file.exists():
        print(f"ERROR: Config file not found at {config_file}")
        sys.exit(1)
    
    test_collision_at_state(config_file)
