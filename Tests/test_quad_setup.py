"""Quick test to verify QuadRestrictionTest setup is correct."""
import sys
import numpy as np
from pathlib import Path

# Add project paths
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir / "src"))
sys.path.append(str(root_dir / "dependencies" / "gncpy" / "src"))

from gncpy.dynamics.aircraft.simple_multirotor_quat import SimpleMultirotorQuat, v_smap_quat
import gncpy.math as gmath
import fcl
from Utils.GeometryUtils import DCM3D
from ConfigurationSpace.ConfigSpace3D import ConfigurationSpace3D
from ConfigurationSpace.Obstacles import StaticObstacle
from Vehicles.Vehicle import Vehicle

print("="*80)
print("QUAD RESTRICTION TEST SETUP VALIDATION")
print("="*80)

# Configuration setup
dim = [0, 10, -2.5, 2.5, -5, 0]  # xMin, xMax, yMin, yMax, zMin, zMax (NED)
config_space = ConfigurationSpace3D(dim)
print(f"\nConfigSpace created: {dim}")

# Build wall with slit
slit_width = 0.25
slit_height = 1
wall_thickness = 1
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

# Add wall obstacles
left_geom = fcl.Box(wall_thickness, upper_y_height, total_z)
left_tf = fcl.Transform(np.eye(3), [x_center, dim[2] + 0.5 * upper_y_height, z_mid])
config_space.add_obstacle(StaticObstacle(left_geom, left_tf))

right_geom = fcl.Box(wall_thickness, upper_y_height, total_z)
right_tf = fcl.Transform(np.eye(3), [x_center, dim[3] - 0.5 * upper_y_height, z_mid])
config_space.add_obstacle(StaticObstacle(right_geom, right_tf))

bottom_geom = fcl.Box(wall_thickness, slit_width, lower_z_height)
bottom_tf = fcl.Transform(np.eye(3), [x_center, y_mid, z_mid - 0.5 * slit_height - 0.5 * lower_z_height])
config_space.add_obstacle(StaticObstacle(bottom_geom, bottom_tf))

top_geom = fcl.Box(wall_thickness, slit_width, upper_z_height)
top_tf = fcl.Transform(np.eye(3), [x_center, y_mid, z_mid + 0.5 * slit_height + 0.5 * upper_z_height])
config_space.add_obstacle(StaticObstacle(top_geom, top_tf))

print(f"Wall with slit created: {config_space.get_num_obstacles()} obstacles")
print(f"  Slit dimensions: {slit_width}m wide x {slit_height}m tall")
print(f"  Slit center: x={x_center}, y={y_mid}, z={z_mid}")

# Initialize dynamics
config_file = Path(__file__).parent / "MPCoptimization" / "SmallQuadrotor.yaml"
QuadDynamics = SimpleMultirotorQuat(str(config_file), effector=None)

# Initial conditions
INIT_POS = np.array([1.5, -1.25, -1.5])  # NED (m)
INIT_VEL = np.array([0.0, 0.0, 0.0])
INIT_EULER = np.array([0.0, 0.0, 0.0])
INIT_RATES = np.array([0.0, 0.0, 0.0])
REF_LAT, REF_LON, TERRAIN_ALT = 40.0, -111.0, 1387.0
ned_mag = np.array([20.0, 5.0, 45.0])

QuadDynamics.set_initial_conditions(
    INIT_POS, INIT_VEL, INIT_EULER, INIT_RATES, REF_LAT, REF_LON, TERRAIN_ALT, ned_mag
)
QuadDynamics.vehicle.takenoff = True

x0 = QuadDynamics.vehicle.state.copy()
print(f"\nQuad dynamics initialized")
print(f"  State length: {len(x0)}")
print(f"  Initial NED position: {x0[v_smap_quat.ned_pos]}")

# Goal
goal_position = np.array([8.5, 1.25, -3.5])
x_goal = x0.copy()
x_goal[v_smap_quat.ned_pos] = goal_position
x_goal[v_smap_quat.body_vel] = np.zeros(3)
x_goal[v_smap_quat.body_rot_rate] = np.zeros(3)

print(f"  Goal position: {goal_position}")

# Create vehicle
vehicle_geometry = fcl.Box(0.3, 0.3, 0.1)
vehicle = Vehicle(
    dynamics_class=QuadDynamics,
    geometry=vehicle_geometry,
    initial_state=x0,
    state_indices={'position': [4, 5, 6]}
)

# Update collision object with rotation
quat = x0[v_smap_quat.quat]
roll, pitch, yaw = gmath.quat_to_euler(quat)
Rx = DCM3D(roll, "x")
Ry = DCM3D(pitch, "y")
Rz = DCM3D(yaw, "z")
dcm = Rz @ Ry @ Rx
ned_pos = x0[v_smap_quat.ned_pos]
vehicle.collision_object.setTransform(fcl.Transform(dcm, ned_pos))

print(f"Vehicle created with geometry: {vehicle_geometry.side}")

# Check initial collision
collision_result = config_space.query_collision_detailed(vehicle.collision_object)
print(f"\nInitial state collision check:")
print(f"  has_collision: {collision_result.has_collision}")
print(f"  is_out_of_bounds: {collision_result.is_out_of_bounds}")
print(f"  min_obstacle_distance: {collision_result.min_obstacle_distance:.4f}m")

if collision_result.has_collision:
    print(f"\n ERROR: Initial state has collision!")
    sys.exit(1)

print(f"\nSetup validation PASSED - ready for optimization!")
print("="*80)
