# Spectral Density Module Updates

## Summary
Updated `spectral_density.py` to match the implementation in `t1_anisotropy_analysis.py`.

## Changes Made

### 1. Added Moving Average Function
```python
def moving_average(data, window_size, axis=0)
```
- Uses convolution with uniform kernel for smoothing
- Can operate on any axis of multi-dimensional arrays
- `mode='same'` preserves array size

### 2. DC Offset Removal
**Before FFT:**
```python
# Calculate DC offset as average of last 100 points
n_offset_points = min(100, len(acf))
dc_offset = np.mean(acf[-n_offset_points:])
acf_corrected = acf - dc_offset
```

### 3. Spectral Density Smoothing
**After FFT:**
```python
# Apply moving average smoothing (default window=5)
if smoothing_window > 1:
    spectral_density = moving_average(spectral_density, smoothing_window, axis=0)
```

### 4. Updated Configuration
Added new parameter to `config.py`:
```python
smoothing_window: int = 5  # Moving average window size
```

## Processing Pipeline

The spectral density calculation now follows this sequence:

1. **Remove DC offset** from ACF (average of last 100 points)
2. **Apply zero-filling** (optional, controlled by `zero_fill_factor`)
3. **Calculate FFT** → spectral density J(ω)
4. **Apply moving average smoothing** (controlled by `smoothing_window`)
5. **Calculate frequency markers** at specific frequencies

## Configuration Parameters

```python
config = NMRConfig(
    zero_fill_factor=2,      # 2× zero-padding for better frequency resolution
    smoothing_window=5,       # 5-point moving average for smoothing
    ...
)
```

## Comparison with t1_anisotropy_analysis.py

### Similarities
✓ DC offset removal (mean of last 100 points)
✓ Zero-padding (2× default)
✓ Moving average smoothing (5-point window)
✓ Same FFT calculation: `J = 2 × dt × Re[FFT(ACF)]`

### Differences
- `t1_anisotropy_analysis.py`: Hardcoded parameters
- `spectral_density.py`: Configurable via NMRConfig

## Testing

Test results show proper functionality:
```
DC offset (avg of last 100 points): 6.22e-01
After zero-fill: 2000 points
Applied moving average smoothing (window=5)
✓ Calculated spectral density
```

## Usage Example

```python
from nmr_calculator.config import NMRConfig
from nmr_calculator.spectral_density import SpectralDensityCalculator

# Configure with smoothing
config = NMRConfig(
    zero_fill_factor=2,
    smoothing_window=5,
    verbose=True
)

# Calculate spectral density
sd_calc = SpectralDensityCalculator(config)
J, frequencies = sd_calc.calculate(acf, time_lags)
```

## Benefits

1. **Reduced noise**: Moving average smoothing reduces high-frequency noise
2. **Better baseline**: DC offset removal improves low-frequency accuracy
3. **Consistent methodology**: Matches validated t1_anisotropy_analysis.py approach
4. **Configurable**: Easy to adjust smoothing and zero-filling parameters
