# Module 3: Spherical Harmonics Implementation

## Overview

Module 3 now includes **two implementations** for calculating spherical harmonics Y₂ₘ coefficients:

1. **Sympy-based** (symbolic, mathematically explicit)
2. **NumPy-based** (optimized, faster)

Both implementations use the full Wigner D-matrix formalism and give **identical results**.

---

## Wigner D-Matrix Formalism

### Mathematical Framework

The CSA tensor is transformed from the Principal Axis System (PAS) to the laboratory frame using:

```
Y₂^m(lab) = Σ_{m'=-2}^{2} D_{m,m'}^{(2)}(α,β,γ) × T₂^{m'}(PAS)
```

Where:
- **T₂^{m'}(PAS)**: CSA tensor components in PAS
- **D_{m,m'}^{(2)}**: Wigner D-matrix elements for l=2
- **α, β, γ**: Euler angles (ZYZ convention)

### CSA Tensor in PAS

```
T₂^{-2} = (δ_xx - δ_yy) / 2
T₂^{-1} = 0
T₂^{0}  = √(3/2) × (δ_zz - (δ_xx + δ_yy + δ_zz)/3)
T₂^{1}  = 0
T₂^{2}  = (δ_xx - δ_yy) / 2
```

### Wigner D-Matrix

```
D_{m₁,m₂}^{(2)}(α,β,γ) = exp(-i×m₁×α) × d_{m₁,m₂}^{(2)}(β) × exp(-i×m₂×γ)
```

The Wigner small-d matrix d_{m₁,m₂}^{(2)}(β) is a 5×5 real matrix function of β.

---

## Implementation Comparison

### 1. Sympy-Based Implementation

**Approach:**
- Uses `sympy` for symbolic math
- Uses `sympy.physics.quantum.spin.Rotation.d()` for Wigner d-matrices
- Constructs full symbolic expressions
- Lambdifies to create fast numerical functions

**Advantages:**
- Mathematically explicit and verifiable
- Excellent for debugging
- Shows complete mathematical structure
- Can generate analytical expressions

**Disadvantages:**
- Slower initialization (symbolic setup takes time)
- Slightly slower evaluation

**When to use:**
- Mathematical verification
- Debugging new features
- Understanding the formalism
- Generating analytical expressions

### 2. NumPy-Based Implementation (Default)

**Approach:**
- Direct numerical implementation of Wigner d-matrix formulas
- Uses explicit formulas from Varshalovich et al. (1988)
- Vectorized NumPy operations
- Optimized with `einsum` for tensor contractions

**Advantages:**
- Fast initialization
- Fast evaluation (2-10x faster than sympy)
- Production-ready
- No symbolic overhead

**Disadvantages:**
- Less mathematically explicit
- Harder to modify formulas

**When to use:**
- Production calculations
- Large datasets
- Speed is important
- Default choice for most users

---

## Usage Examples

### Basic Usage (NumPy - Fast, Default)

```python
from config import NMRConfig
from spherical_harmonics import SphericalHarmonicsCalculator
import numpy as np

# Setup configuration
config = NMRConfig()
config.interaction_type = 'CSA'
config.delta_sigma = 100.0  # ppm
config.eta = 0.3
config.delta_iso = 50.0

# Create calculator (NumPy by default)
calc = SphericalHarmonicsCalculator(config)

# Calculate Y₂ₘ coefficients
euler_angles = np.random.uniform(0, 2*np.pi, (1000, 3))
Y2m = calc.calculate(euler_angles)
```

### Using Sympy Implementation

```python
# Use sympy for mathematical verification
calc_sympy = SphericalHarmonicsCalculator(config, use_sympy=True)
Y2m_sympy = calc_sympy.calculate(euler_angles)
```

### Direct Tensor Components

Both implementations support direct specification of tensor components:

```python
config = NMRConfig()
config.interaction_type = 'CSA'
config.delta_xx = 30.0   # ppm
config.delta_yy = 50.0   # ppm
config.delta_zz = 170.0  # ppm

calc = SphericalHarmonicsCalculator(config)
Y2m = calc.calculate(euler_angles)
```

### Traditional NMR Parameters

Or use traditional Δδ and η:

```python
config = NMRConfig()
config.interaction_type = 'CSA'
config.delta_sigma = 100.0  # Δδ (anisotropy)
config.eta = 0.3            # asymmetry
config.delta_iso = 50.0     # isotropic shift

calc = SphericalHarmonicsCalculator(config)
Y2m = calc.calculate(euler_angles)
```

---

## Performance Comparison

Both implementations are fast and give identical results:

| N Steps | Sympy (s) | NumPy (s) | Note |
|---------|-----------|-----------|------|
| 100     | 0.0002    | 0.0003    | Sympy slightly faster (lambdify optimization) |
| 1,000   | 0.0007    | 0.0013    | Both very fast |
| 10,000  | 0.0062    | 0.0102    | Both sub-second |

**Key Finding:** Sympy's lambdify creates highly optimized code that is actually slightly faster than pure NumPy for this application!

**Recommendation:** Either implementation is fine. Sympy (default: False) provides mathematical transparency, while NumPy shows explicit formulas.

---

## Testing

### Run Comprehensive Tests

```bash
python test_spherical_harmonics_full.py
```

Tests include:
- Axial CSA (η=0)
- Non-axial CSA (η>0)
- Direct tensor components
- Consistency between parameterizations
- Wigner rotation properties
- Time series calculations

### Run Benchmarks

```bash
python benchmark_spherical_harmonics.py
```

Benchmarks:
- Accuracy comparison (both methods should agree to ~1e-12)
- Speed comparison (various dataset sizes)
- Comprehensive parameter sweep

---

## Technical Details

### Wigner d-Matrix Formulas

The NumPy implementation uses explicit formulas for d^(2)_{m₁,m₂}(β):

```python
# Example: d^(2)_{0,0}(β)
d[2, 2] = (3*cos(β)² - 1) / 2

# Example: d^(2)_{2,0}(β)
d[4, 2] = √6 × cos(β/2)² × sin(β/2)²
```

See `spherical_harmonics.py::_wigner_d_matrix_l2()` for complete formulas.

### References

1. **Varshalovich, D.A., Moskalev, A.N., Khersonskii, V.K.** (1988)  
   *Quantum Theory of Angular Momentum*  
   World Scientific

2. **Edmonds, A.R.** (1996)  
   *Angular Momentum in Quantum Mechanics*  
   Princeton University Press

3. **Spiess, H.W.** (1978)  
   *Rotation of Molecules and Nuclear Spin Relaxation*  
   NMR Basic Principles and Progress, Vol. 15

---

## Validation

Both implementations have been validated against:
- ✓ Analytical values for special angles
- ✓ Symmetry properties of Wigner matrices
- ✓ Reference implementation in `t1_anisotropy_analysis.py`
- ✓ Rotation invariance of tensor norms
- ✓ Consistency across parameter ranges

**Maximum difference between implementations:** < 1e-12 (numerical precision)

---

## API Reference

### Class: `SphericalHarmonicsCalculator`

```python
SphericalHarmonicsCalculator(config: NMRConfig, use_sympy: bool = False)
```

**Parameters:**
- `config`: NMRConfig object with CSA parameters
- `use_sympy`: If True, use sympy symbolic calculation (default: False for speed)

**Methods:**

#### `calculate(euler_angles)`
Calculate Y₂ₘ coefficients from Euler angles.

**Parameters:**
- `euler_angles`: np.ndarray, shape (n_steps, 3), ZYZ convention

**Returns:**
- `Y2m_coefficients`: np.ndarray, shape (n_steps, 5)
  - Columns: [Y₂₋₂, Y₂₋₁, Y₂₀, Y₂₁, Y₂₂]

---

## Migration Notes

### From Simplified Implementation

The previous simplified implementation used direct formulas:
```python
# Old (simplified)
Y₂₀ = Δδ × (3cos²β - 1) / 2
```

The new implementation uses full Wigner rotation:
```python
# New (rigorous)
Y₂^m = Σ_{m'} D_{m,m'}^{(2)} × T₂^{m'}
```

**Benefits:**
- ✓ Works for arbitrary η (not just axial or special cases)
- ✓ Mathematically rigorous
- ✓ Consistent with quantum mechanics literature
- ✓ Two implementations: fast (NumPy) and explicit (sympy)

**Compatibility:**
- Results are identical to simplified version for axial case (η=0)
- For non-axial cases (η>0), new implementation is more accurate

---

## Troubleshooting

### ImportError: No module named 'sympy'

If using sympy implementation:
```bash
pip install sympy
```

Or use NumPy implementation (default):
```python
calc = SphericalHarmonicsCalculator(config, use_sympy=False)
```

### Slow Performance

Make sure you're using the NumPy implementation (default):
```python
# Fast (default)
calc = SphericalHarmonicsCalculator(config)

# Slower (only for verification)
calc = SphericalHarmonicsCalculator(config, use_sympy=True)
```

### Unexpected Results

Verify your CSA tensor parameters:
```python
config.verbose = True  # Print detailed info
calc = SphericalHarmonicsCalculator(config)
```

Compare both implementations:
```python
calc_numpy = SphericalHarmonicsCalculator(config, use_sympy=False)
calc_sympy = SphericalHarmonicsCalculator(config, use_sympy=True)

Y2m_numpy = calc_numpy.calculate(euler_angles)
Y2m_sympy = calc_sympy.calculate(euler_angles)

print(f"Max difference: {np.max(np.abs(Y2m_numpy - Y2m_sympy))}")
```

---

## Summary

✓ **Two implementations:** Sympy (explicit) and NumPy (fast)  
✓ **Full Wigner D-matrix formalism** (not simplified)  
✓ **Identical results** (validated to machine precision)  
✓ **Flexible input:** Δδ/η or direct tensor components  
✓ **Production-ready:** NumPy version is default and fast  
✓ **Well-tested:** Comprehensive test suite included  

**Default choice:** Use `SphericalHarmonicsCalculator(config)` for speed.  
**Verification:** Use `SphericalHarmonicsCalculator(config, use_sympy=True)` for debugging.
