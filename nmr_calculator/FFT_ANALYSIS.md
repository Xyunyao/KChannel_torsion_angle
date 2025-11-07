# FFT vs Direct Method Analysis

## Key Question
**Does FFT (without DC removal) match the reference?**

## Answer: YES, but only with zero-padding! ✓

---

## Test Results

### Method Comparison

| Method | Match with Reference | Max Difference |
|--------|---------------------|----------------|
| **Direct loop** | ✓ YES | 0.0e+00 |
| **FFT (no zero-pad, no DC)** | ✗ NO | ~0.02-0.19 |
| **FFT + zero-pad (no DC)** | ✓ YES | ~1e-16 |

### Detailed Results

**Signal: Exp decay + DC + noise**
```
Direct vs Reference:           0.0e+00 ✓
FFT (no ZF, no DC) vs Ref:    1.94e-01 ✗
FFT+ZF (no DC) vs Ref:        4.44e-16 ✓
```

**Signal: Pure noise**
```
Direct vs Reference:           0.0e+00 ✓
FFT (no ZF, no DC) vs Ref:    1.80e-02 ✗
FFT+ZF (no DC) vs Ref:        2.22e-16 ✓
```

**Signal: Pure exponential**
```
Direct vs Reference:           0.0e+00 ✓
FFT (no ZF, no DC) vs Ref:    2.14e-02 ✗
FFT+ZF (no DC) vs Ref:        8.33e-17 ✓
```

---

## Conclusion

### The Problem Was Two-Fold:

1. **DC offset removal** ← WRONG (we fixed this)
2. **Missing zero-padding** ← Also important!

### Why Zero-Padding Matters

Without zero-padding, FFT has **circular convolution** artifacts:
- FFT assumes signal is periodic
- Without padding, end wraps to beginning
- Creates artificial correlations at long lags

With zero-padding:
- Eliminates wrap-around artifacts
- Gives same result as direct method
- Matches reference exactly (diff < 1e-15)

---

## Three Valid Implementations

### Option 1: Direct Method (Current)
```python
def _calculate_acf_direct(series, max_lag, lag_step):
    corr = []
    for tau in range(0, max_lag, lag_step):
        if tau == 0:
            val = np.mean(series * np.conj(series))
        else:
            val = np.mean(series[:-tau] * np.conj(series[tau:]))
        corr.append(val)
    return np.array(corr)
```
- ✓ Exact match with reference
- ✓ Simple and clear
- ✗ Slower (~5ms for 10k steps)

### Option 2: FFT with Zero-Padding
```python
def _calculate_acf_fft_with_zeropad(series, max_lag, lag_step):
    n = len(series)
    
    # Zero-pad to 2× length (minimum for avoiding circular artifacts)
    n_padded = 2 * n
    series_padded = np.concatenate([series, np.zeros(n_padded - n)])
    
    # FFT
    fft_series = np.fft.fft(series_padded)
    power_spectrum = fft_series * np.conj(fft_series)
    acf_full = np.fft.ifft(power_spectrum).real
    
    # Take first n points
    acf = acf_full[:n]
    
    # Normalize by overlap counts
    overlap_counts = np.arange(n, 0, -1)
    acf /= overlap_counts
    
    # Subsample
    return acf[0:max_lag:lag_step]
```
- ✓ Exact match with reference (diff ~ 1e-16)
- ✓ Faster (~2ms for 10k steps)
- ✗ More complex
- ✗ Requires zero-padding

### Option 3: FFT without Zero-Padding
```python
def _calculate_acf_fft_no_zeropad(series, max_lag, lag_step):
    # ... same as option 2 but without zero-padding ...
```
- ✗ Does NOT match reference (diff ~ 0.02-0.19)
- ✗ Circular convolution artifacts

---

## Recommendation

**Current implementation (Direct method) is CORRECT** ✓

We could also use **FFT with zero-padding** (Option 2), which would be:
- Faster (2.5× speedup)
- Exact match with reference
- But more complex

However, since:
1. Direct method is already fast enough (~5ms)
2. Direct method is clearer and easier to verify
3. We don't need maximum speed in ACF calculation

**Recommendation: Keep the direct method** ✓

---

## Why Did We Think FFT Was Wrong?

**Original issue**: FFT with DC removal did NOT match reference

We tested:
```python
# Old method
dc_offset = np.mean(series[-100:])
series_centered = series - dc_offset
# Then FFT with zero-padding
# Result: Large error (~0.68) ✗
```

Now we know:
- **DC removal was the problem** (81.9% error)
- **FFT itself is fine** (with zero-padding)

---

## Summary

| Implementation | DC Removal | Zero-Pad | Match Ref | Speed |
|----------------|-----------|----------|-----------|-------|
| Reference | No | N/A | - | Direct |
| Direct (current) | No | N/A | ✓ 0.0 | ~5ms |
| FFT + ZF | No | Yes | ✓ 1e-16 | ~2ms |
| FFT (no ZF) | No | No | ✗ 0.02 | ~2ms |
| FFT + DC + ZF | Yes | Yes | ✗ 0.68 | ~2ms |

**Best choice**: Direct method (current implementation) ✓

- Simple
- Correct
- Fast enough
- Matches reference exactly

---

## If We Wanted to Use FFT

If performance becomes critical, we could switch to FFT with zero-padding:

```python
def _calculate_acf_fft(self, series, max_lag, lag_step):
    """
    FFT-based ACF calculation.
    
    Requires 2× zero-padding to avoid circular convolution artifacts.
    Matches reference implementation exactly.
    """
    n = len(series)
    
    # NO DC removal
    
    # Zero-pad to 2× length (required for correct normalization)
    n_padded = 2 * n
    series_padded = np.concatenate([series, np.zeros(n_padded - n)])
    
    # Power spectrum
    fft_series = np.fft.fft(series_padded)
    power_spectrum = fft_series * np.conj(fft_series)
    
    # Inverse FFT
    acf_full = np.fft.ifft(power_spectrum).real
    
    # Take first n points
    acf = acf_full[:n]
    
    # Normalize by overlap counts
    overlap_counts = np.arange(n, 0, -1)
    acf /= overlap_counts
    
    # Subsample
    return acf[0:max_lag:lag_step]
```

This would give:
- ✓ Exact match with reference (diff ~ 1e-16)
- ✓ 2.5× faster
- But adds complexity

---

**Current Status**: Direct method is correct and fast enough. No change needed unless performance becomes critical.
