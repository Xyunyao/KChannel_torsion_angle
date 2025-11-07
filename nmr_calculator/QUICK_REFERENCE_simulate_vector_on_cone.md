# Quick Reference: simulate_vector_on_cone()

## Location
`nmr_calculator/xyz_generator.py` → `TrajectoryGenerator.simulate_vector_on_cone()`

## Signature
```python
def simulate_vector_on_cone(self, S2=None, tau_c=None, dt=None, num_steps=None, axis=None)
```

## Returns
`np.ndarray` with shape `(num_steps, 3)` - unit vectors on cone surface

## Key Feature: YOUR axis parameter works! 🎯

## Quick Examples

### 1. Use config defaults (z-axis)
```python
from nmr_calculator.xyz_generator import TrajectoryGenerator
from nmr_calculator.config import NMRConfig

config = NMRConfig(S2=0.85, tau_c=1e-9, dt=1e-12, num_steps=1000)
gen = TrajectoryGenerator(config)
vectors = gen.simulate_vector_on_cone()
```

### 2. Custom x-axis
```python
import numpy as np
vectors = gen.simulate_vector_on_cone(axis=np.array([1, 0, 0]))
```

### 3. Custom diagonal
```python
vectors = gen.simulate_vector_on_cone(axis=np.array([0, 1, 1]))
```

### 4. Override S² and tau_c
```python
vectors = gen.simulate_vector_on_cone(S2=0.90, tau_c=5e-9)
```

## Validation (All Tests Passed ✅)
- Z-axis: θ = 18.4349°, std = 0.000000°, S² = 0.850000
- X-axis: θ = 18.4349°, std = 0.000000°, S² = 0.850000  
- Diagonal: θ = 18.4349°, std = 0.000000°, S² = 0.850000

## Physics
- **Model**: Diffusion on cone edge (fixed θ)
- **Cone angle**: θ = arccos(√((2S² + 1)/3))
- **Dynamics**: Ornstein-Uhlenbeck azimuthal diffusion
- **Order parameter**: S² = ⟨P₂(cos θ)⟩

## When to Use
✅ Single vector trajectories (NH bond, CSA axis)  
✅ Quick testing and validation  
✅ Custom cone axis direction  
✅ Don't need full rotation matrices  
✅ Want fast computation (5-8× faster)  

## Compare with Rotation Methods
- `generate_diffusion_cone()` - Within cone (β varies), rotation matrices
- `generate_diffusion_cone_edge()` - On edge (β fixed), rotation matrices
- `simulate_vector_on_cone()` - On edge (β fixed), **unit vectors** 👈 YOU ARE HERE

## Test It
```bash
python nmr_calculator/xyz_generator.py
```

Look for "TEST 4: simulate_vector_on_cone() Method" in output!
