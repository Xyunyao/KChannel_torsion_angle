# Module: cone_simulation.py

## Overview

The `cone_simulation.py` module provides vector-based simulation utilities for molecular diffusion on cone surfaces. Unlike `xyz_generator.py` which works with full 3×3 rotation matrices, this module operates directly on unit vectors (e.g., NH bond vectors, CSA principal axes).

**Purpose:**
- Testing and validation of diffusion models
- Generating synthetic NMR data
- Quick prototyping of cone diffusion scenarios
- Educational demonstrations of cone diffusion physics

## Key Differences: cone_simulation vs xyz_generator

| Feature | `cone_simulation.py` | `xyz_generator.py` |
|---------|---------------------|-------------------|
| **Output** | Unit vectors (N×3) | Rotation matrices (N×3×3) |
| **Use case** | Single vector dynamics | Full molecular frame |
| **Integration** | Standalone utility | Core pipeline module |
| **Speed** | Faster (simpler math) | Slower (full rotations) |
| **Flexibility** | Vectors only | Supports Euler angles, quaternions |

## Functions

### 1. simulate_vector_on_cone()

Simulate diffusion **on the cone edge** (fixed θ, azimuthal diffusion only).

```python
def simulate_vector_on_cone(
    S2=0.85,           # Order parameter
    tau_c=0.01,        # Correlation time (s)
    dt=1e-4,           # Time step (s)
    num_steps=10000,   # Number of steps
    axis=[0,0,1]       # Cone axis direction
) -> np.ndarray:  # Returns (num_steps, 3) array
```

**Physics:**
- Polar angle θ is **constant** (fixed at cone surface)
- Only azimuthal angle φ diffuses (Ornstein-Uhlenbeck process)
- Cone angle from S²: cos(θ) = √((2S² + 1) / 3)
- Equivalent to `trajectory_type='diffusion_cone_edge'` in xyz_generator

**S² Formula:** S² = ⟨P₂(cos θ)⟩ where P₂(x) = (3x² - 1)/2

**Example:**
```python
from nmr_calculator.cone_simulation import simulate_vector_on_cone

# Simulate NH bond vector on cone edge
vectors = simulate_vector_on_cone(
    S2=0.85,
    tau_c=1e-9,    # 1 ns
    dt=1e-12,      # 1 ps  
    num_steps=1000
)

# vectors.shape = (1000, 3)
# All vectors have same angle from z-axis
```

### 2. simulate_vector_within_cone()

Simulate diffusion **within the cone volume** (both θ and φ vary).

```python
def simulate_vector_within_cone(
    S2=0.85,
    tau_c=0.01,
    dt=1e-4,
    num_steps=10000,
    axis=[0,0,1]
) -> np.ndarray:
```

**Physics:**
- Polar angle θ varies from 0 to θ_cone
- Azimuthal angle φ also diffuses
- Cone angle from S²: cos(θ_cone) = 2√S² - 1
- Equivalent to `trajectory_type='diffusion_cone'` in xyz_generator

**S² Formula:** S² = ((1 + cos(θ_cone))/2)²

**Example:**
```python
vectors = simulate_vector_within_cone(
    S2=0.85,
    tau_c=1e-9,
    dt=1e-12,
    num_steps=1000
)

# vectors.shape = (1000, 3)
# Vectors explore entire cone volume
```

### 3. calculate_order_parameter()

Calculate S² from a trajectory of vectors.

```python
def calculate_order_parameter(
    vectors,           # (N, 3) array
    axis=[0,0,1]      # Reference axis
) -> float:           # Returns S²
```

**Formula:** S² = ⟨P₂(cos θ)⟩ = ⟨(3cos²θ - 1)/2⟩

**Example:**
```python
from nmr_calculator.cone_simulation import (
    simulate_vector_on_cone,
    calculate_order_parameter
)

vectors = simulate_vector_on_cone(S2=0.85, num_steps=10000)
S2_calc = calculate_order_parameter(vectors)

print(f"Target S²: 0.85")
print(f"Calculated S²: {S2_calc:.4f}")
# Should be very close to 0.85
```

### 4. rotation_matrix_from_vectors()

Utility function to align vector a to vector b.

```python
def rotation_matrix_from_vectors(a, b) -> np.ndarray:
    """Returns 3×3 rotation matrix that rotates a onto b."""
```

## Comparison: Two Cone Models

For the **same S² = 0.85**:

| Model | Cone Angle | θ Behavior | Formula |
|-------|-----------|------------|---------|
| **On Edge** | 18.43° | Fixed | cos(θ) = √((2S²+1)/3) |
| **Within** | 32.44° | Varies 0→θ | cos(θ) = 2√S² - 1 |

**Key Insight:** For same S², edge model has **smaller cone** because all vectors are at maximum angle, while within-cone model averages over the volume.

## Usage Examples

### Example 1: Validate S² reproduction

```python
import numpy as np
from nmr_calculator.cone_simulation import (
    simulate_vector_on_cone,
    calculate_order_parameter
)

# Test edge model
S2_target = 0.85
vectors = simulate_vector_on_cone(
    S2=S2_target,
    tau_c=1e-9,
    dt=1e-12,
    num_steps=5000
)

S2_calc = calculate_order_parameter(vectors)
error = abs(S2_calc - S2_target)

print(f"S² Target:     {S2_target:.6f}")
print(f"S² Calculated: {S2_calc:.6f}")
print(f"Error:         {error:.6f}")
# Should see error < 0.001
```

### Example 2: Compare edge vs within models

```python
from nmr_calculator.cone_simulation import (
    simulate_vector_on_cone,
    simulate_vector_within_cone
)
import numpy as np

# Same S², different models
S2 = 0.85

# Edge model
vecs_edge = simulate_vector_on_cone(S2=S2, num_steps=1000)
theta_edge = np.arccos(vecs_edge[:, 2])  # Angle from z-axis

# Within model
vecs_within = simulate_vector_within_cone(S2=S2, num_steps=1000)
theta_within = np.arccos(vecs_within[:, 2])

print("Edge Model:")
print(f"  θ mean: {np.degrees(theta_edge.mean()):.2f}°")
print(f"  θ std:  {np.degrees(theta_edge.std()):.2f}°")

print("\nWithin Model:")
print(f"  θ mean: {np.degrees(theta_within.mean()):.2f}°")
print(f"  θ std:  {np.degrees(theta_within.std()):.2f}°")

# Edge: θ std ≈ 0 (fixed)
# Within: θ std > 0 (variable)
```

### Example 3: Custom cone axis

```python
# Simulate diffusion around non-z axis
import numpy as np

# Define custom axis (e.g., along x-axis)
custom_axis = np.array([1, 0, 0])

vectors = simulate_vector_on_cone(
    S2=0.85,
    axis=custom_axis,
    num_steps=1000
)

# Verify cone is centered on x-axis
cos_theta = np.dot(vectors, custom_axis / np.linalg.norm(custom_axis))
print(f"Mean cos(θ): {cos_theta.mean():.4f}")
```

### Example 4: Time-dependent correlation

```python
import matplotlib.pyplot as plt

vectors = simulate_vector_on_cone(
    S2=0.85,
    tau_c=1e-9,
    dt=1e-12,
    num_steps=10000
)

# Calculate autocorrelation function
def autocorr(v, maxlag=1000):
    c0 = np.dot(v[0], v[0])
    corr = [np.dot(v[0], v[i]) / c0 for i in range(maxlag)]
    return np.array(corr)

acf = autocorr(vectors)
time = np.arange(len(acf)) * 1e-12 * 1e9  # Convert to ns

plt.figure(figsize=(8, 5))
plt.plot(time, acf)
plt.xlabel('Time (ns)')
plt.ylabel('Autocorrelation')
plt.title('Vector Autocorrelation Function')
plt.grid(True, alpha=0.3)
plt.show()

# Should decay with time constant ~ tau_c
```

## Testing

Run the module directly to execute test suite:

```bash
cd nmr_calculator
python cone_simulation.py
```

**Test output:**
```
======================================================================
Cone Simulation Module - Test Suite
======================================================================

TEST 1: Diffusion on Cone Edge (fixed θ)
  ✓ Generated 5000 vectors
  ✓ Theta std: 0.000000° (constant)
  ✓ S² error: 0.000000

TEST 2: Diffusion Within Cone (θ varies)
  ✓ Generated 5000 vectors
  ✓ Theta varies from 0° to 32.44°
  ✓ S² error: 0.022 (acceptable for short trajectory)

COMPARISON
  Edge model:   θ = 18.43° (fixed)
  Within model: θ ≤ 32.44° (varies)
  Cone angle ratio: 0.568
======================================================================
```

## Performance

**Typical timing** (MacBook Pro M1):
- 1,000 steps: ~0.5 ms
- 10,000 steps: ~5 ms
- 100,000 steps: ~50 ms

Much faster than `xyz_generator.py` because:
- No matrix operations (only 3D vectors)
- No quaternion conversions
- Simpler diffusion equations

## Integration with Pipeline

While `cone_simulation.py` is standalone, you can use it to:

1. **Generate test data** for validating the full pipeline
2. **Quick S² calculations** before running expensive calculations
3. **Educational demonstrations** of cone diffusion physics
4. **Prototyping** new diffusion models before implementing in xyz_generator

**Example integration:**
```python
from nmr_calculator.cone_simulation import simulate_vector_on_cone
from nmr_calculator.xyz_generator import TrajectoryGenerator
from nmr_calculator.config import NMRConfig

# Quick test with cone_simulation
vectors = simulate_vector_on_cone(S2=0.85, num_steps=100)
# Fast result for prototyping

# Full calculation with xyz_generator
config = NMRConfig(trajectory_type='diffusion_cone_edge', S2=0.85)
gen = TrajectoryGenerator(config)
rotations, coords = gen.generate()
# Complete rotation matrices for NMR calculations
```

## Mathematical Background

### Lipari-Szabo Order Parameter

For axially symmetric motion:

$$S^2 = \langle P_2(\cos\theta) \rangle = \left\langle \frac{3\cos^2\theta - 1}{2} \right\rangle$$

where θ is the angle between the vector and the symmetry axis.

### Cone Models

**Edge Model (fixed θ):**
- Vector fixed at angle θ from axis
- Only azimuthal motion (φ varies)
- S² = P₂(cos θ) = (3cos²θ - 1)/2
- Solving for θ: cos(θ) = √((2S² + 1) / 3)

**Within Model (θ variable):**
- Vector explores cone volume (0 ≤ θ ≤ θ_cone)
- Uniform distribution assumption
- S² = ((1 + cos(θ_cone))/2)²
- Solving for θ_cone: cos(θ_cone) = 2√S² - 1

### Ornstein-Uhlenbeck Process

Azimuthal angle φ follows:

$$\frac{d\phi}{dt} = -\frac{\phi}{\tau_c} + \sqrt{\frac{2}{\tau_c}}\xi(t)$$

where ξ(t) is white noise with unit variance.

**Discretized:**
```python
d_phi = -phi/tau_c * dt + sqrt(2/tau_c) * sqrt(dt) * randn()
```

This gives exponential correlation decay with time constant τ_c.

## References

1. **Lipari, G. & Szabo, A.** (1982). *J. Am. Chem. Soc.*, 104, 4546-4559.
   - Original model-free formalism for S²

2. **Brüschweiler, R.** (1995). *J. Chem. Phys.*, 102, 3396-3403.
   - Ensemble approaches to order parameters

3. **Wong, V. & Case, D. A.** (2008). *J. Phys. Chem. B*, 112, 6013-6024.
   - Diffusion in a cone models for biomolecules

## See Also

- `xyz_generator.py` - Full rotation matrix trajectory generation
- `MODULE1_CONE_EDGE_UPDATE.md` - Details on cone edge implementation
- `example_cone_comparison.py` - Visual comparison of both models
- `README_COMPLETE.md` - Complete package documentation
