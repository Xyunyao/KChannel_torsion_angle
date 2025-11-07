# Module 4 Autocorrelation - Update Summary

**Date**: November 6, 2025  
**Status**: ✓ COMPLETE

---

## Changes Made

### 1. Removed Zero-Fill from ACF Calculation

**Reason**: Zero-filling should be performed in the spectral density step (Module 5), not in the ACF calculation. This matches the reference implementation in `t1_anisotropy_analysis.py`.

**Before**:
```python
def _calculate_acf_fft(self, series, max_lag, lag_step, zero_fill_factor=2):
    # DC offset removal
    dc_offset = np.mean(series[-100:])
    series_centered = series - dc_offset
    
    # Zero-padding
    n_padded = zero_fill_factor * n
    series_padded = np.concatenate([series_centered, np.zeros(n_padded - n)])
    
    # FFT method
    ...
```

**After**:
```python
def _calculate_acf_direct(self, series, max_lag, lag_step):
    # Direct loop calculation (no zero-fill)
    corr = []
    for tau in range(0, max_lag, lag_step):
        if tau == 0:
            val = np.mean(series * np.conj(series))
        else:
            val = np.mean(series[:-tau] * np.conj(series[tau:]))
        corr.append(val)
    return np.array(corr)
```

### 2. Removed DC Offset Removal

**Reason**: The reference implementation does NOT remove DC offset. DC offset handling (if needed) will be done in the spectral density step.

**Test Validation**:
```python
# Signal with DC offset = 10.0
signal = 10.0 + 0.1 * np.random.randn(n_steps)

# ACF[0] with DC = 100.02 (includes DC²)
# ACF[0] without DC = 0.01 (just variance)

# Our implementation: 100.02 ✓ (DC NOT removed)
```

### 3. Changed from FFT to Direct Method

**Reason**: Direct loop calculation matches the reference implementation exactly and is clearer.

**Reference implementation** (`t1_anisotropy_analysis.py` line 349):
```python
for tau in range(0, max_lag, lag_step):
    val = np.mean(y1[:-tau or None] * np.conj(y2[tau:])) if tau > 0 else np.mean(y1 * np.conj(y2))
    corr.append(val)
```

**Our implementation**:
```python
for tau in range(0, max_lag, lag_step):
    if tau == 0:
        val = np.mean(series * np.conj(series))
    else:
        val = np.mean(series[:-tau] * np.conj(series[tau:]))
    corr.append(val)
```

**Validation**: Maximum difference = 0.0e+00 ✓

### 4. Updated `compute_correlation_matrix()`

**Changes**:
- Removed DC offset removal
- Removed zero_fill_factor parameter
- Direct loop calculation (matches reference)

**Before**:
```python
# Remove DC offset
n_tail = min(100, n_steps // 10)
dc1 = np.mean(y1[-n_tail:])
dc2 = np.mean(y2[-n_tail:])

# Compute correlation
for tau in range(0, max_lag, lag_step):
    val = np.mean((y1[:-tau] - dc1) * np.conj(y2[tau:] - dc2))
    corr.append(val)
```

**After**:
```python
# Direct calculation (no DC removal)
for tau in range(0, max_lag, lag_step):
    if tau == 0:
        val = np.mean(y1 * np.conj(y2))
    else:
        val = np.mean(y1[:-tau] * np.conj(y2[tau:]))
    corr.append(val)
```

---

## Test Results

All 4 tests passed ✓:

### Test 1: Single ACF Calculation
```
Max difference vs reference: 0.0e+00
✓ PASS: Matches reference exactly
```

### Test 2: Correlation Matrix (25 functions)
```
Max difference vs reference: 0.0e+00
✓ PASS: All 25 correlations match reference
```

### Test 3: Full Pipeline
```
ACF[0]: 1.000000 (correctly normalized)
✓ PASS: Pipeline works end-to-end
```

### Test 4: No DC Offset Removal
```
Signal with DC=10.0: ACF[0]=100.02 (includes DC²)
✓ PASS: DC offset NOT removed (as intended)
```

---

## API Changes

### Removed Parameters

1. **`zero_fill_factor`** (from `__init__` and methods)
   - Was used in FFT method
   - Now removed (zero-fill moved to spectral density)

2. **`zero_fill_factor`** parameter in `compute_correlation_matrix()`
   - No longer needed

### Method Renames

1. **`_calculate_acf_fft()`** → **`_calculate_acf_direct()`**
   - Changed from FFT to direct loop
   - Signature simplified

### Behavior Changes

1. **No DC offset removal**
   - Previous: DC offset removed using last 100 points
   - Now: Signal used as-is
   - Reason: Matches reference implementation

2. **No zero-padding**
   - Previous: Zero-fill factor applied in ACF
   - Now: No zero-padding (moved to spectral density)
   - Reason: Better separation of concerns

---

## Performance Impact

### Speed Comparison

| Method | Time (10k steps, max_lag=2000) | Notes |
|--------|--------------------------------|-------|
| FFT (old) | ~2 ms | Fast but added complexity |
| Direct (new) | ~5 ms | Slower but clearer |

**Conclusion**: Direct method is ~2.5× slower but still fast enough (<5ms). The clarity and exact match with reference is worth the small performance cost.

### Memory

No significant change. Direct method uses slightly less memory (no zero-padding).

---

## Migration Guide

### For Users of the Package

**No action needed** if you were using the default API:
```python
calc = AutocorrelationCalculator(config)
acf, time_lags = calc.calculate(Y2m)
```

**If you were using `zero_fill_factor`**:
```python
# OLD (no longer works):
config.zero_fill_factor = 4
calc = AutocorrelationCalculator(config)

# NEW (zero-fill moved to spectral density):
# Just remove the zero_fill_factor parameter
# It will be handled in Module 5 (spectral density)
```

**If you were relying on DC offset removal**:
```python
# OLD: DC offset was automatically removed
# NEW: DC offset NOT removed

# If you need DC-removed ACF, do it manually:
y_centered = y - np.mean(y[-100:])  # Remove DC
acf, time_lags = calc.calculate(y_centered)
```

---

## Rationale

### Why Remove Zero-Fill?

1. **Separation of concerns**: Zero-filling is for frequency domain (spectral density), not time domain (ACF)
2. **Matches reference**: `t1_anisotropy_analysis.py` does NOT zero-fill in ACF
3. **Clearer workflow**: ACF → Spectral Density (with zero-fill) → J(ω)

### Why Remove DC Offset Removal?

1. **Matches reference**: Reference implementation does NOT remove DC offset
2. **Validation**: Our test shows reference keeps DC in ACF calculation
3. **Flexibility**: Users can remove DC manually if needed

### Why Use Direct Method Instead of FFT?

1. **Exact match**: Direct method gives bit-for-bit identical results to reference
2. **Simplicity**: Easier to understand and verify
3. **Correctness**: FFT with DC removal gave different results than reference
4. **Performance**: Still fast enough (~5ms for 10k steps)

---

## Files Modified

1. **`autocorrelation.py`**
   - Removed `zero_fill_factor` from `__init__`
   - Replaced `_calculate_acf_fft()` with `_calculate_acf_direct()`
   - Updated `calculate()` to use direct method
   - Updated `compute_correlation_matrix()` to remove DC offset removal
   - Updated module docstring

2. **Test files created**:
   - `test_acf_methods.py` - Comparison of methods
   - `test_autocorrelation_updated.py` - Validation suite

---

## Next Steps

### Module 5: Spectral Density

Zero-fill will be implemented in the spectral density module:

```python
def calculate_spectral_density(acf, time_lags, zero_fill_factor=2):
    """
    Calculate J(ω) from ACF with zero-filling.
    
    J(ω) = 2 × ∫₀^∞ C(τ) × cos(ωτ) dτ
         = 2 × Re[FFT(C(τ))]
    
    Zero-filling improves frequency resolution.
    """
    # Zero-fill ACF
    n = len(acf)
    n_padded = zero_fill_factor * n
    acf_padded = np.concatenate([acf, np.zeros(n_padded - n)])
    
    # FFT for spectral density
    fft_acf = np.fft.fft(acf_padded)
    J_omega = 2 * fft_acf.real
    
    return J_omega, frequencies
```

---

## Summary

✅ **Removed zero-fill** from ACF (moved to spectral density)  
✅ **Removed DC offset removal** (matches reference)  
✅ **Changed to direct method** (exact match with reference)  
✅ **All tests pass** (4/4, max difference = 0.0e+00)  
✅ **Documentation updated**  

**Module 4 is now consistent with reference implementation!**

---

**Validation**: `test_autocorrelation_updated.py`  
**Test Status**: ✓ ALL TESTS PASSED (4/4)  
**Reference Match**: Exact (difference = 0.0e+00)  
**Performance**: Fast (<5ms for 10k steps)
