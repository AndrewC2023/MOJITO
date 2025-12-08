"""Test saturation limits in input functions."""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add project source to path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir / "src"))

from Controls.NBBMPC import PiecewiseConstantInput, LinearInterpolationInput, SplineInterpolationInput

# Test with keyframes that would violate saturation limits
num_keyframes = 5
control_dim = 2
start_time = 0.0
end_time = 10.0

# Keyframes with values outside [0.2, 0.8] range
keyframe_values = np.array([
    [0.1, 0.9],   # Below min, above max
    [0.5, 0.5],   # Within limits
    [0.0, 1.0],   # At extreme bounds
    [0.3, 0.7],   # Within limits
    [0.15, 0.85]  # Below min, above max
])

print("Testing saturation limits")
print(f"Keyframe values:\n{keyframe_values}")
print(f"\nSaturation limits: u_min=0.2, u_max=0.8")

# Test with scalar saturation limits
spline_scalar = SplineInterpolationInput(
    num_keyframes, totalSteps=100, control_dim=control_dim,
    u_min=0.2, u_max=0.8
)
spline_scalar.updateStartAndEndTimes(start_time, end_time)
spline_scalar.updateKeyFrameValues(keyframe_values)

# Test with per-dimension saturation limits
u_min_vec = np.array([0.2, 0.3])
u_max_vec = np.array([0.8, 0.9])

linear_vector = LinearInterpolationInput(
    num_keyframes, totalSteps=100, control_dim=control_dim,
    u_min=u_min_vec, u_max=u_max_vec
)
linear_vector.updateStartAndEndTimes(start_time, end_time)
linear_vector.updateKeyFrameValues(keyframe_values)

print(f"\nVector saturation: u_min={u_min_vec}, u_max={u_max_vec}")

# Sample at many points
times = np.linspace(start_time, end_time, 200)
spline_controls = np.array([spline_scalar.calculateInput(t) for t in times])
linear_controls = np.array([linear_vector.calculateInput(t) for t in times])

# Verify saturation
print("\n--- Verification ---")
print(f"Spline (scalar limits):")
print(f"  Control 0 range: [{spline_controls[:, 0].min():.3f}, {spline_controls[:, 0].max():.3f}]")
print(f"  Expected: [0.200, 0.800]")
print(f"  Control 1 range: [{spline_controls[:, 1].min():.3f}, {spline_controls[:, 1].max():.3f}]")
print(f"  Expected: [0.200, 0.800]")

print(f"\nLinear (vector limits):")
print(f"  Control 0 range: [{linear_controls[:, 0].min():.3f}, {linear_controls[:, 0].max():.3f}]")
print(f"  Expected: [0.200, 0.800]")
print(f"  Control 1 range: [{linear_controls[:, 1].min():.3f}, {linear_controls[:, 1].max():.3f}]")
print(f"  Expected: [0.300, 0.900]")

# Check that values are properly clamped
assert spline_controls[:, 0].min() >= 0.2 - 1e-10, "Control 0 violates min limit"
assert spline_controls[:, 0].max() <= 0.8 + 1e-10, "Control 0 violates max limit"
assert spline_controls[:, 1].min() >= 0.2 - 1e-10, "Control 1 violates min limit"
assert spline_controls[:, 1].max() <= 0.8 + 1e-10, "Control 1 violates max limit"

assert linear_controls[:, 0].min() >= 0.2 - 1e-10, "Control 0 violates min limit"
assert linear_controls[:, 0].max() <= 0.8 + 1e-10, "Control 0 violates max limit"
assert linear_controls[:, 1].min() >= 0.3 - 1e-10, "Control 1 violates min limit"
assert linear_controls[:, 1].max() <= 0.9 + 1e-10, "Control 1 violates max limit"

print("\n All saturation limits respected!")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
keyframe_times = np.linspace(start_time, end_time, num_keyframes)

for dim in range(control_dim):
    ax = axes[dim]
    
    # Plot interpolated controls
    ax.plot(times, spline_controls[:, dim], 'r-', linewidth=2, label='Spline (scalar limits)', alpha=0.8)
    ax.plot(times, linear_controls[:, dim], 'g-', linewidth=2, label='Linear (vector limits)', alpha=0.8)
    
    # Plot keyframes (raw values, before saturation)
    ax.scatter(keyframe_times, keyframe_values[:, dim], s=100, c='black', 
               marker='o', zorder=10, label='Keyframes (raw)', alpha=0.5)
    
    # Plot saturation limits
    ax.axhline(0.2, color='blue', linestyle='--', linewidth=1.5, label='Scalar limits', alpha=0.7)
    ax.axhline(0.8, color='blue', linestyle='--', linewidth=1.5, alpha=0.7)
    
    if dim == 1:
        # Show different limits for dimension 1
        ax.axhline(0.3, color='orange', linestyle=':', linewidth=1.5, label='Vector limits', alpha=0.7)
        ax.axhline(0.9, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel(f'Control {dim+1}', fontsize=11)
    ax.set_title(f'Control Dimension {dim+1} with Saturation', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    ax.set_xlim([start_time, end_time])
    ax.set_ylim([0, 1])

plt.tight_layout()
out_path = Path(__file__).parent / "saturation_limits_test.png"
plt.savefig(out_path, dpi=150)
plt.close()

print(f"\n Saved saturation test plot to {out_path}")
print("\n All tests passed!")
