# Wigner D-Matrix Formula Fix

## Date
January 2025

## Issue Summary
The Wigner reduced d-matrix calculation in `rotation_matrix.py` had formula errors that violated unitarity at certain angles.

## Bugs Identified

### Bug 1: Wrong coefficient signs for m=±1 diagonal elements (β=0)
- **Location**: `_wigner_d_small_rank2()` method, lines 150 and 174
- **Problem**: At β=0 (identity rotation), d²₋₁,₋₁(0) and d²₁,₁(0) returned 3 instead of 1
- **Root cause**: Wrong signs in formulas
  ```python
  # BEFORE (WRONG):
  d[1, 1] = c2**2 * (2 * cb + 1)  # At β=0: 1² × (2×1 + 1) = 3 ✗
  d[3, 3] = c2**2 * (2 * cb + 1)  # At β=0: 1² × (2×1 + 1) = 3 ✗
  
  # AFTER (CORRECT):
  d[1, 1] = c2**2 * (2 * cb - 1)  # At β=0: 1² × (2×1 - 1) = 1 ✓
  d[3, 3] = c2**2 * (2 * cb - 1)  # At β=0: 1² × (2×1 - 1) = 1 ✓
  ```

### Bug 2: Wrong sign for d²₁,₋₁(β) (β=π/2)
- **Location**: `_wigner_d_small_rank2()` method, line 154
- **Problem**: At β=π/2, d²₁,₋₁(π/2) returned -0.5 instead of +0.5
- **Root cause**: Extra negative sign in formula
  ```python
  # BEFORE (WRONG):
  d[3, 1] = -s2**2 * (2 * cb + 1)  # At β=π/2: -0.5 × 1 = -0.5 ✗
  
  # AFTER (CORRECT):
  d[3, 1] = s2**2 * (2 * cb + 1)   # At β=π/2: 0.5 × 1 = +0.5 ✓
  ```

### Related fix: Symmetry partner d²₋₁,₁(β)
- **Location**: Line 170
- **Fix**: Changed sign to maintain symmetry
  ```python
  # BEFORE:
  d[1, 3] = s2**2 * (2 * cb - 1)
  
  # AFTER:
  d[1, 3] = s2**2 * (2 * cb + 1)
  ```

## Verification

### Test Results BEFORE Fix
- **Identity rotation**: D(0,0,0) = diag([1, 3, 1, 3, 1]) ✗
- **90° rotation**: Max |D×D† - I| = 0.5 ✗
- **Frobenius norm**: Error up to 4.68 ✗

### Test Results AFTER Fix
- **Identity rotation**: D(0,0,0) = I exactly ✓
- **90° rotation**: Max |D×D† - I| = 1.8×10⁻¹⁶ (machine precision) ✓
- **Frobenius norm**: Error < 6×10⁻¹⁶ (machine precision) ✓
- **All 12 tests**: 7 functional + 5 analytical = 12/12 PASSED ✓

## Mathematical Validation

The fixes ensure that the reduced Wigner d-matrix satisfies:
1. **Identity**: d²(0) = I (5×5 identity matrix)
2. **Unitarity**: d²(β) × d²(β)ᵀ = I for all β
3. **Symmetry**: Proper relations between d²ₘ,ₘ'(β) elements
4. **Reference match**: Agrees with Varshalovich Table 4.3 at β=π/2

## Files Modified
1. `rotation_matrix.py`: Lines 150, 154, 170, 174 in `_wigner_d_small_rank2()`
2. `test_rotated_correlation_analytical.py`: Removed "known issue" warnings from Tests 4 and 5

## Testing
All analytical properties now verified to machine precision:
- ✓ Isotropic averaging (off-diagonal suppression)
- ✓ Decay rate preservation (unitarity)
- ✓ Hermiticity preservation
- ✓ Frobenius norm preservation (unitarity)
- ✓ Identity rotation (D(0,0,0) = I)

## Reference
Varshalovich, D. A., Moskalev, A. N., & Khersonskii, V. K. (1988). 
*Quantum Theory of Angular Momentum*. World Scientific.
- Table 4.3: d²ₘₘ'(β) matrix elements
