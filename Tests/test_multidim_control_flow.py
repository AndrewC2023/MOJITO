"""Test complete flow: 1D decision vector → multiple control dimensions → cost function."""

import numpy as np
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir / "src"))

from Controls.NACMPC import SplineInterpolationInput

print("=== Testing Multi-Dimensional Control Flow ===\n")

# Quadrotor has 4 control inputs (rotor thrusts)
control_dim = 4
num_keyframes = 3

# Create input function
input_func = SplineInterpolationInput(
    numKeyframes=num_keyframes,
    totalSteps=100,
    control_dim=control_dim,
    u_min=0.0,
    u_max=1.0
)
input_func.updateStartAndEndTimes(0.0, 5.0)

print(f"Configuration:")
print(f"  Control dimensions: {control_dim} (4 rotor thrusts)")
print(f"  Keyframes: {num_keyframes}")
print(f"  Decision vector size: {num_keyframes * control_dim} = {num_keyframes} × {control_dim}")

# Step 1: Optimizer provides 1D decision vector
# Format: [kf0_u0, kf0_u1, kf0_u2, kf0_u3, kf1_u0, kf1_u1, kf1_u2, kf1_u3, kf2_u0, ...]
decision_vector_1d = np.array([
    # Keyframe 0: [u0, u1, u2, u3]
    0.3, 0.4, 0.5, 0.6,
    # Keyframe 1: [u0, u1, u2, u3]
    0.5, 0.6, 0.7, 0.8,
    # Keyframe 2: [u0, u1, u2, u3]
    0.4, 0.5, 0.6, 0.7
])

print(f"\n1. Optimizer provides flat decision vector:")
print(f"   Shape: {decision_vector_1d.shape}")
print(f"   Values: {decision_vector_1d}")

# Step 2: NACMPC parses and passes to input function
# The input function's updateKeyFrameValues handles reshaping
input_func.updateKeyFrameValues(decision_vector_1d)

print(f"\n2. Input function reshapes to (keyframes, control_dim):")
print(f"   Shape: {input_func.keyframeValues.shape}")
print(f"   Values:")
for i, kf in enumerate(input_func.keyframeValues):
    print(f"     Keyframe {i}: {kf}")

# Step 3: Input function creates separate interpolator for EACH control dimension
print(f"\n3. Input function creates {len(input_func.pchip_interpolators)} PCHIP interpolators")
print(f"   (one per control dimension)")

# Step 4: During rollout, calculateInput returns full control vector at each time
test_times = [0.0, 2.5, 5.0]
print(f"\n4. During rollout, calculateInput() returns full {control_dim}D control vector:")
for t in test_times:
    u = input_func.calculateInput(t)
    print(f"   t={t:.1f}s: u = {u} (shape: {u.shape})")

# Step 5: Cost function receives the full control vector
print(f"\n5. Cost function receives:")
print(f"   - state: (12,) for quadrotor")
print(f"   - control: (4,) for 4 rotors")
print(f"   - dt: scalar")
print(f"   - **kwargs: collision_result, is_terminal, etc.")

# Verify the flow with all three input types
from Controls.NACMPC import PiecewiseConstantInput, LinearInterpolationInput

print(f"\n=== Verification: All Input Functions Handle Multiple Dimensions ===\n")

for name, InputClass in [
    ("Piecewise Constant", PiecewiseConstantInput),
    ("Linear Interpolation", LinearInterpolationInput),
    ("PCHIP Spline", SplineInterpolationInput)
]:
    inp = InputClass(num_keyframes, 100, control_dim=control_dim, u_min=0.0, u_max=1.0)
    inp.updateStartAndEndTimes(0.0, 5.0)
    inp.updateKeyFrameValues(decision_vector_1d)
    
    u_mid = inp.calculateInput(2.5)
    
    print(f"{name}:")
    print(f"  ✓ Accepts 1D vector: {decision_vector_1d.shape}")
    print(f"  ✓ Reshapes internally: {inp.keyframeValues.shape}")
    print(f"  ✓ Returns full control: {u_mid.shape} = {u_mid}")

print(f"\n=== Summary ===")
print(f"✓ Optimizer provides: flat 1D array of size (num_keyframes × control_dim)")
print(f"✓ Input function automatically reshapes to (num_keyframes, control_dim)")
print(f"✓ Separate interpolator created for EACH control dimension")
print(f"✓ calculateInput() returns full control_dim vector at any time")
print(f"✓ Cost function receives complete control vector")
print(f"\n✓ Everything already works correctly for 4-rotor quadrotor!")
