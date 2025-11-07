# Module 4 Autocorrelation - Dual Method Implementation

**Date**: November 6, 2025  
**Status**: ✓ PRODUCTION READY

---

## Overview

Module 4 now supports **two calculation methods** for autocorrelation:

1. **Direct method** (default) - Loop-based, exact match with reference
2. **FFT method** (optional) - ~100× faster for large datasets

Both methods give **identical results** to machine precision (~1e-16 difference).

---

## Method Comparison

### Performance (Validated)

| Dataset Size | Direct Method | FFT Method | Speedup |
|--------------|---------------|------------|---------|
| 5,000 steps | 99 ms | 0.9 ms | **106×** |
| 10,000 steps | 250 ms | 2.3 ms | **109×** |
| 20,000 steps | 528 ms | 4.7 ms | **113×** |
| 50,000 steps | 1,317 ms | 18.3 ms | **72×** |

### Accuracy (Validated)

| Method | vs Reference | vs Each Other |
|--------|-------------|---------------|
| Direct | 0.0e+00 ✓ | - |
| FFT | 8.3e-17 ✓ | 3.3e-16 ✓ |

**Both methods match reference and each other to machine precision!**

---

## Usage

### Default: Direct Method

```python
from config import NMRConfig
from autocorrelation import AutocorrelationCalculator

config = NMRConfig(
    num_steps=10000,
    max_lag=2000,
    # ... other parameters ...
)

# Default: uses direct method
calc = AutocorrelationCalculator(config)
acf, time_lags = calc.calculate(Y2m)
```

**Use when:**
- Default choice (simple and clear)
- Dataset < 20k steps
- Maximum accuracy verification needed

### Fast: FFT Method

```python
# Use FFT for speed
calc = AutocorrelationCalculator(config, use_fft=True)
acf, time_lags = calc.calculate(Y2m)
```

**Use when:**
- Dataset > 20k steps
- Speed is critical
- Running many calculations

**Performance gain**: ~100× faster!

---

## Implementation Details

### Direct Method

```python
def _calculate_acf_direct(series, max_lag, lag_step):
    """Loop-based calculation."""
    corr = []
    for tau in range(0, max_lag, lag_step):
        if tau == 0:
            val = np.mean(series * np.conj(series))
        else:
            val = np.mean(series[:-tau] * np.conj(series[tau:]))
        corr.append(val)
    return np.array(corr)
```

**Features:**
- Exact match with `t1_anisotropy_analysis.py` reference
- No DC offset removal
- No zero-padding needed
- O(N × M) complexity where N=length, M=max_lag

### FFT Method

```python
def _calculate_acf_fft(series, max_lag, lag_step):
    """FFT-based calculation with 2× zero-padding."""
    n = len(series)
    
    # Zero-pad to 2× length (REQUIRED!)
    n_padded = 2 * n
    series_padded = np.concatenate([series, np.zeros(n_padded - n)])
    
    # Power spectrum
    fft_series = np.fft.fft(series_padded)
    power_spectrum = fft_series * np.conj(fft_series)
    
    # Autocorrelation via IFFT
    acf_full = np.fft.ifft(power_spectrum).real
    acf = acf_full[:n]
    
    # Normalize by overlap counts
    overlap_counts = np.arange(n, 0, -1)
    acf /= overlap_counts
    
    return acf[0:max_lag:lag_step]
```

**Features:**
- Uses Wiener-Khinchin theorem: ACF = IFFT(|FFT(x)|²)
- 2× zero-padding **required** (without it: ~2-19% error!)
- No DC offset removal
- O(N log N) complexity

---

## Why 2× Zero-Padding is Required

**Test results:**

| Zero-Padding | Difference from Reference |
|--------------|---------------------------|
| **None** | 2-19% error ✗ |
| **2× length** | ~1e-16 (perfect) ✓ |

**Reason**: Without zero-padding, FFT has circular convolution artifacts:
- FFT assumes signal is periodic
- End wraps to beginning
- Creates false correlations at long lags

With 2× zero-padding:
- Eliminates wrap-around
- Gives true linear convolution
- Matches direct method exactly

---

## Validation

### Test Suite: `test_dual_acf_methods.py`

**All 5 tests PASSED** ✓

```
Test 1: Method Equivalence
  Max difference: 3.3e-16 ✓

Test 2: Reference Match
  Direct vs Ref: 0.0e+00 ✓
  FFT vs Ref: 8.3e-17 ✓

Test 3: Performance
  FFT is 72-113× faster ✓

Test 4: Correlation Matrix
  Max difference: 0.0e+00 ✓

Test 5: Edge Cases
  All cases match ✓
```

### Edge Cases Tested

- Constant signal ✓
- Zero signal ✓
- Delta function ✓
- Large DC offset ✓

All cases: Methods match to machine precision!

---

## When to Use Each Method

### Use Direct Method (default) When:

✓ Dataset < 20k steps (fast enough)  
✓ Learning/understanding the code  
✓ Maximum clarity needed  
✓ Reference validation critical  

### Use FFT Method (use_fft=True) When:

✓ Dataset > 20k steps (significant speedup)  
✓ Running many calculations (batch processing)  
✓ Speed is critical  
✓ After validating it matches direct method  

---

## Examples

### Basic Usage

```python
# Small dataset: use default (direct)
config_small = NMRConfig(num_steps=10000, max_lag=2000)
calc = AutocorrelationCalculator(config_small)
acf, time_lags = calc.calculate(Y2m)
```

### Large Dataset

```python
# Large dataset: use FFT
config_large = NMRConfig(num_steps=100000, max_lag=20000)
calc = AutocorrelationCalculator(config_large, use_fft=True)
acf, time_lags = calc.calculate(Y2m)
```

### Comparing Methods

```python
# Verify both methods give same result
calc_direct = AutocorrelationCalculator(config, use_fft=False)
calc_fft = AutocorrelationCalculator(config, use_fft=True)

acf_direct, _ = calc_direct.calculate(Y2m)
acf_fft, _ = calc_fft.calculate(Y2m)

diff = np.max(np.abs(acf_direct - acf_fft))
print(f"Difference: {diff:.2e}")  # Should be ~1e-16
```

### Batch Processing (Use FFT!)

```python
# Process many trajectories
trajectories = [...]  # List of 100 trajectories

calc = AutocorrelationCalculator(config, use_fft=True)  # Use FFT!

acfs = []
for Y2m in trajectories:
    acf, _ = calc.calculate(Y2m)
    acfs.append(acf)

# With FFT: ~10× faster than direct method
```

---

## Technical Notes

### Why FFT is So Fast

Direct method: O(N × M)
- N = series length (e.g., 10,000)
- M = max_lag (e.g., 2,000)
- Operations: 10,000 × 2,000 = 20 million

FFT method: O(N log N)
- With zero-padding: N = 20,000
- FFT ops: 20,000 × log₂(20,000) ≈ 280,000

**Speedup**: 20M / 280k ≈ **71×** theoretical (matches observed!)

### Memory Usage

| Method | Memory | Notes |
|--------|--------|-------|
| Direct | 1× | Minimal |
| FFT | 2× | Due to zero-padding |

For 10k steps with complex128:
- Direct: ~80 KB
- FFT: ~160 KB

Memory difference is negligible for modern systems.

---

## API Reference

### AutocorrelationCalculator

```python
class AutocorrelationCalculator:
    def __init__(self, config: NMRConfig, use_fft: bool = False):
        """
        Initialize autocorrelation calculator.
        
        Parameters
        ----------
        config : NMRConfig
            Configuration object
        use_fft : bool, optional
            If True, use FFT method (faster for large datasets)
            If False, use direct method (default, clearer)
        """
```

### Methods

```python
# Calculate ACF
acf, time_lags = calc.calculate(Y2m_coefficients)

# Correlation matrix (always uses direct method)
corr_matrix = calc.compute_correlation_matrix(Y2m_coefficients)

# Internal methods (for testing/validation)
acf = calc._calculate_acf_direct(series, max_lag, lag_step)
acf = calc._calculate_acf_fft(series, max_lag, lag_step)
```

---

## Recommendations

### For Most Users

```python
# Use default (direct) - simple and clear
calc = AutocorrelationCalculator(config)
```

### For Production/Large Datasets

```python
# Use FFT for speed
calc = AutocorrelationCalculator(config, use_fft=True)

# But validate once:
calc_check = AutocorrelationCalculator(config, use_fft=False)
assert np.allclose(acf_fft, acf_direct, atol=1e-12)
```

### For Package Developers

```python
# Test both methods match
import pytest

def test_methods_match():
    calc_direct = AutocorrelationCalculator(config, use_fft=False)
    calc_fft = AutocorrelationCalculator(config, use_fft=True)
    
    acf_direct, _ = calc_direct.calculate(Y2m)
    acf_fft, _ = calc_fft.calculate(Y2m)
    
    assert np.allclose(acf_direct, acf_fft, atol=1e-12)
```

---

## Summary

✅ **Two methods available**: Direct (default) and FFT (fast)  
✅ **Both validated**: Identical results to machine precision  
✅ **FFT is ~100× faster**: Use for large datasets (>20k steps)  
✅ **Direct is default**: Clearer, matches reference exactly  
✅ **2× zero-padding required**: For FFT to match direct method  
✅ **All tests pass**: 5/5 validation tests  

**Module 4 is production-ready with dual method support!** ✓

---

**Files:**
- `autocorrelation.py` - Updated with dual method support
- `test_dual_acf_methods.py` - Comprehensive validation (5 tests)
- `FFT_ANALYSIS.md` - Technical analysis
- `MODULE4_DUAL_METHODS.md` - This document

**Test Status**: ✓ ALL TESTS PASSED (5/5)  
**Performance**: FFT 72-113× faster  
**Accuracy**: Both methods identical (~1e-16 difference)
