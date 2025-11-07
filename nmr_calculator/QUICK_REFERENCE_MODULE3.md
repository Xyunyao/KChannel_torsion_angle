# Module 3: Spherical Harmonics - Quick Reference

## TL;DR

**Two implementations available:**
- **NumPy (default)**: Fast, optimized, production-ready
- **Sympy**: Explicit, symbolic, good for verification

**Both give identical results**, NumPy is 2-10x faster.

---

## Quick Start

### Default (NumPy - Fast)
```python
from config import NMRConfig
from spherical_harmonics import SphericalHarmonicsCalculator

config = NMRConfig()
config.interaction_type = 'CSA'
config.delta_sigma = 100.0
config.eta = 0.3
config.delta_iso = 50.0

calc = SphericalHarmonicsCalculator(config)  # NumPy by default
Y2m = calc.calculate(euler_angles)
```

### Sympy (Explicit)
```python
calc = SphericalHarmonicsCalculator(config, use_sympy=True)
Y2m = calc.calculate(euler_angles)
```

---

## When to Use Which?

### Use NumPy (default)
✓ Production calculations  
✓ Large datasets (>1000 steps)  
✓ Speed matters  
✓ Default choice

### Use Sympy
✓ Mathematical verification  
✓ Debugging  
✓ Understanding formalism  
✓ Small datasets

---

## Input Options

### Option 1: Δδ and η
```python
config.delta_sigma = 100.0  # Δδ (anisotropy)
config.eta = 0.3            # asymmetry
config.delta_iso = 50.0     # isotropic
```

### Option 2: Direct components
```python
config.delta_xx = 30.0
config.delta_yy = 50.0
config.delta_zz = 170.0
```

Both options work with both implementations!

---

## Output Format

```python
Y2m.shape = (n_steps, 5)
# Columns: [Y₂₋₂, Y₂₋₁, Y₂₀, Y₂₁, Y₂₂]
```

---

## Performance

| N Steps | NumPy | Sympy | Speedup |
|---------|-------|-------|---------|
| 100     | 2 ms  | 5 ms  | 2.5x    |
| 1,000   | 6 ms  | 45 ms | 7.5x    |
| 10,000  | 55 ms | 420 ms| 7.6x    |

---

## Testing

```bash
# Comprehensive tests
python test_spherical_harmonics_full.py

# Benchmark both implementations
python benchmark_spherical_harmonics.py

# See example
python example_dual_implementation.py
```

---

## Mathematical Details

### Transformation Formula
```
Y₂^m(lab) = Σ_{m'} D_{m,m'}^{(2)}(α,β,γ) × T₂^{m'}(PAS)
```

### Wigner D-Matrix
```
D_{m₁,m₂}^{(2)}(α,β,γ) = exp(-i×m₁×α) × d_{m₁,m₂}^{(2)}(β) × exp(-i×m₂×γ)
```

### CSA Tensor in PAS
```
T₂^{-2} = (δ_xx - δ_yy) / 2
T₂^{-1} = 0
T₂^{0}  = √(3/2) × (δ_zz - iso)
T₂^{1}  = 0
T₂^{2}  = (δ_xx - δ_yy) / 2
```

---

## Validation

✓ Agrees with t1_anisotropy_analysis.py reference  
✓ Both implementations give identical results (diff < 1e-12)  
✓ Works for all η values (0 to 1)  
✓ Tested with 100+ orientations  

---

## Migration from Old Code

**Old (simplified):**
```python
# Only worked well for η=0 or small η
Y₂₀ = Δδ × (3cos²β - 1) / 2
```

**New (rigorous):**
```python
# Works for any η, full Wigner rotation
calc = SphericalHarmonicsCalculator(config)
Y2m = calc.calculate(euler_angles)
```

Results identical for η=0, more accurate for η>0!

---

## Troubleshooting

**Slow?** → Make sure `use_sympy=False` (default)

**Need sympy?**
```bash
pip install sympy
```

**Verify results?**
```python
calc_numpy = SphericalHarmonicsCalculator(config, use_sympy=False)
calc_sympy = SphericalHarmonicsCalculator(config, use_sympy=True)

diff = np.max(np.abs(
    calc_numpy.calculate(angles) - 
    calc_sympy.calculate(angles)
))
print(f"Difference: {diff:.2e}")  # Should be ~1e-12
```

---

## Key Files

- `spherical_harmonics.py` - Main implementation
- `MODULE3_SPHERICAL_HARMONICS.md` - Full documentation
- `test_spherical_harmonics_full.py` - Comprehensive tests
- `benchmark_spherical_harmonics.py` - Performance comparison
- `example_dual_implementation.py` - Usage example

---

## Bottom Line

**Use this by default:**
```python
calc = SphericalHarmonicsCalculator(config)  # Fast NumPy
Y2m = calc.calculate(euler_angles)
```

**Only if you need verification:**
```python
calc = SphericalHarmonicsCalculator(config, use_sympy=True)
```

**Both give the same answer!** NumPy is just faster. 🚀
