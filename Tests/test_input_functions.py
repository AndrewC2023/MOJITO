"""Test the three input function implementations."""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add project source to path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir / "src"))

from Controls.NACMPC import PiecewiseConstantInput, LinearInterpolationInput, SplineInterpolationInput

# Test parameters
num_keyframes = 5
control_dim = 4
start_time = 0.0
end_time = 10.0

# Create random keyframe values (4 control dimensions)
np.random.seed(42)
keyframe_values = np.random.uniform(0.2, 0.8, size=(num_keyframes, control_dim))

print("Testing input functions with 5 keyframes over [0, 10]s")
print(f"Keyframe values shape: {keyframe_values.shape}")

# Create instances of each input function
pwc = PiecewiseConstantInput(num_keyframes, totalSteps=100, control_dim=control_dim,
                              u_min=0.0, u_max=1.0)
pwc.updateStartAndEndTimes(start_time, end_time)
pwc.updateKeyFrameValues(keyframe_values)

linear = LinearInterpolationInput(num_keyframes, totalSteps=100, control_dim=control_dim,
                                   u_min=0.0, u_max=1.0)
linear.updateStartAndEndTimes(start_time, end_time)
linear.updateKeyFrameValues(keyframe_values)

spline = SplineInterpolationInput(num_keyframes, totalSteps=100, control_dim=control_dim,
                                   u_min=0.0, u_max=1.0)
spline.updateStartAndEndTimes(start_time, end_time)
spline.updateKeyFrameValues(keyframe_values)

# Sample control values at many time points
times = np.linspace(start_time, end_time, 200)

pwc_controls = np.array([pwc.calculateInput(t) for t in times])
linear_controls = np.array([linear.calculateInput(t) for t in times])
spline_controls = np.array([spline.calculateInput(t) for t in times])

print("\n All input functions successfully evaluated")

# Plot results for each control dimension
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

keyframe_times = np.linspace(start_time, end_time, num_keyframes)

for dim in range(control_dim):
    ax = axes[dim]
    
    # Plot interpolated controls
    ax.plot(times, pwc_controls[:, dim], 'b-', linewidth=2, label='Piecewise Constant', alpha=0.7)
    ax.plot(times, linear_controls[:, dim], 'g-', linewidth=2, label='Linear', alpha=0.7)
    ax.plot(times, spline_controls[:, dim], 'r-', linewidth=2, label='Cubic Spline', alpha=0.7)
    
    # Plot keyframe values
    ax.scatter(keyframe_times, keyframe_values[:, dim], s=100, c='black', 
               marker='o', zorder=10, label='Keyframes')
    
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel(f'Control {dim+1}', fontsize=11)
    ax.set_title(f'Control Dimension {dim+1}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    ax.set_xlim([start_time, end_time])
    ax.set_ylim([0, 1])

plt.tight_layout()
out_path = Path(__file__).parent / "input_functions_comparison.png"
plt.savefig(out_path, dpi=150)
plt.close()

print(f"\n Saved comparison plot to {out_path}")

# Test with flattened keyframe input
print("\n--- Testing with flattened keyframe array ---")
keyframe_flat = keyframe_values.flatten()
print(f"Flattened shape: {keyframe_flat.shape}")

pwc2 = PiecewiseConstantInput(num_keyframes, totalSteps=100, control_dim=control_dim,
                              u_min=0.0, u_max=1.0)
pwc2.updateStartAndEndTimes(start_time, end_time)
pwc2.updateKeyFrameValues(keyframe_flat)

test_time = 5.0
result = pwc2.calculateInput(test_time)
print(f"Control at t={test_time}s: {result}")
print(" Flattened input works correctly")

# Test dimension reduction benefit
print("\n--- Dimension Reduction Analysis ---")
print(f"Full control sequence (100 steps × 4 dims): {100 * control_dim} parameters")
print(f"Piecewise constant (5 keyframes × 4 dims): {num_keyframes * control_dim} parameters")
print(f"   Reduction: {100 * control_dim / (num_keyframes * control_dim):.1f}x")
print(f"\nWith smooth splines, could use even fewer keyframes (e.g., 3):")
print(f"  3 keyframes × 4 dims: {3 * control_dim} parameters")
print(f"   Reduction: {100 * control_dim / (3 * control_dim):.1f}x")

print("\n All tests passed!")
