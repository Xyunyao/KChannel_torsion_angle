# Module 5: Rotated Correlation Function Matrix

## Overview
Module 5 calculates rotated correlation matrices using Wigner D-matrices for ensemble-averaged NMR relaxation calculations. This accounts for molecular tumbling in solution, which is essential for matching experimental NMR measurements.

## Physics Background

### Rotational Transformation
When a molecule tumbles in solution, the correlation functions transform according to:

```
C'_{m1,m2}(τ) = Σ_{m3,m4} D²_{m1,m3} × C_{m3,m4}(τ) × D²*_{m2,m4}
```

In matrix form:
```
C'(τ) = D × C(τ) × D†
```

where:
- **D**: Wigner D-matrix (5×5) for rank-2 spherical harmonics
- **D†**: Hermitian conjugate of D
- **C(τ)**: Correlation matrix from Module 4
- **C'(τ)**: Rotated correlation matrix

### Ensemble Averaging
For isotropic tumbling in solution:

```
⟨C(τ)⟩ = (1/N) Σᵢ Dᵢ × C(τ) × Dᵢ†
```

This averages over many orientations (typically N=1000-10000) to simulate random tumbling.

## Implementation

### Class: `RotatedCorrelationCalculator`

Main class for rotating correlation matrices.

#### Initialization
```python
from nmr_calculator import RotatedCorrelationCalculator, NMRConfig

config = NMRConfig(verbose=True)
calc = RotatedCorrelationCalculator(config)
```

#### Method 1: Load Pre-computed Wigner D Library
**Recommended for large ensembles** (faster, memory efficient)

```python
# Load library (e.g., 5000 pre-computed orientations)
calc.load_wigner_d_library('path/to/wigner_d_order2_N5000.npz')

# Rotate correlation matrix
rotated_corrs = calc.rotate_correlation_matrix(corr_matrix)
# Output shape: (5000, 5, 5, n_lags)

# Ensemble average
ensemble_avg = calc.compute_ensemble_average()
# Output shape: (5, 5, n_lags)
```

#### Method 2: Compute from Euler Angles
**Useful for specific MD trajectories**

```python
# Compute Wigner D from Euler angles
euler_angles = np.array([[α₁, β₁, γ₁], [α₂, β₂, γ₂], ...])  # (n_orientations, 3)
calc.compute_wigner_d_from_euler(euler_angles)

# Rotate and average
rotated_corrs = calc.rotate_correlation_matrix(corr_matrix)
ensemble_avg = calc.compute_ensemble_average()
```

### Function: `rotate_all()`

Standalone function matching reference implementation:

```python
from nmr_calculator import rotate_all

# Convert correlation dict to array
m_values = [-2, -1, 0, 1, 2]
A = np.array([corr_matrix[(m1, m2)] for m1 in m_values for m2 in m_values])
A = A.reshape(5, 5, -1)  # Shape: (5, 5, n_lags)

# Rotate
rotated_corrs = rotate_all(D_2_lib, A)  # Shape: (n_orientations, 5, 5, n_lags)

# Extract specific component for T1 calculation
acf_m1 = rotated_corrs[:, 3, 3, :]  # m=1, index 3 (m=-2,-1,0,1,2 → indices 0,1,2,3,4)
```

## Key Features

### 1. Dual Mode Operation
- **Load pre-computed library**: Fast, memory efficient for large ensembles
- **Compute from Euler angles**: Flexible for MD trajectory analysis

### 2. Optional Saving
Save individual rotated matrices for later analysis:

```python
rotated_corrs = calc.rotate_correlation_matrix(
    corr_matrix,
    save_individual=True,
    save_dir='./rotated_matrices'
)
# Saves: rotated_matrices/rotated_corr_orientation_{i:05d}.npz
```

Save ensemble average:

```python
ensemble_avg = calc.compute_ensemble_average(
    save_path='./ensemble_avg.npz'
)
```

### 3. Load Saved Results
Skip expensive calculation by loading pre-computed results:

```python
calc = RotatedCorrelationCalculator(config)
ensemble_avg = calc.load_ensemble_average('./ensemble_avg.npz')
```

### 4. Optimized Performance
- **Numba acceleration**: Automatic if numba is installed
- **Parallel processing**: Uses `prange` for multi-core
- **Fallback**: Pure numpy if numba unavailable

```python
# With numba (5000 orientations, 2000 lags):
#   ~2-5 seconds
# Without numba:
#   ~30-60 seconds
```

## Usage Patterns

### Pattern 1: Complete Pipeline
```python
from nmr_calculator import (
    NMRConfig, 
    TrajectoryGenerator,
    EulerConverter,
    SphericalHarmonicsCalculator,
    AutocorrelationCalculator,
    RotatedCorrelationCalculator
)

# Configuration
config = NMRConfig(
    trajectory_type='diffusion_cone',
    S2=0.85,
    tau_c=5e-9,
    dt=0.02e-9,
    num_steps=5000,
    interaction_type='CSA',
    delta_sigma=100.0,
    max_lag=1000,
    verbose=True
)

# Generate trajectory
gen = TrajectoryGenerator(config)
rotations, vectors = gen.generate()

# Euler angles
converter = EulerConverter(config)
euler_angles = converter.convert(rotations=rotations)

# Y2m coefficients
sph_calc = SphericalHarmonicsCalculator(config)
Y2m = sph_calc.calculate(vectors)

# Correlation matrix
acf_calc = AutocorrelationCalculator(config)
corr_matrix = acf_calc.compute_correlation_matrix(Y2m)

# Rotated correlations
rot_calc = RotatedCorrelationCalculator(config)
rot_calc.compute_wigner_d_from_euler(euler_angles)
rotated_corrs = rot_calc.rotate_correlation_matrix(corr_matrix)
ensemble_avg = rot_calc.compute_ensemble_average()
```

### Pattern 2: Reference-Style Usage
Matches `t1_anisotropy_analysis.py`:

```python
# Load pre-computed library
Wig_D2_lib = np.load('wigner_d_order2_N5000.npz', allow_pickle=True)
D_2_lib = Wig_D2_lib['d2_matrix']  # Shape: (5000, 5, 5)

# Convert correlation matrix to array
A = np.array([corr_matrix[(m1, m2)] for m1 in range(-2, 3) for m2 in range(-2, 3)])
A = A.reshape(5, 5, -1)

# Rotate
rotated_corrs = rotate_all(D_2_lib, A)

# Extract component for spectral density
extracted_acf_m1 = rotated_corrs[:, 3, 3, :]  # m=1 component
```

### Pattern 3: Save/Resume Workflow
For long calculations, save intermediate results:

```python
# First run: compute and save
calc = RotatedCorrelationCalculator(config)
calc.load_wigner_d_library('wigner_lib.npz')
rotated_corrs = calc.rotate_correlation_matrix(
    corr_matrix,
    save_individual=True,
    save_dir='./individual_matrices'
)
ensemble_avg = calc.compute_ensemble_average(
    save_path='./ensemble_avg.npz'
)

# Later runs: just load
calc2 = RotatedCorrelationCalculator(config)
ensemble_avg = calc2.load_ensemble_average('./ensemble_avg.npz')
# Continue with spectral density calculation...
```

## Data Formats

### Wigner D-matrix Library
Pre-computed library file (`.npz` format):

```python
# Structure:
{
    'd2_matrix': np.ndarray,  # Shape: (n_orientations, 5, 5), dtype: complex128
    # Optional metadata...
}

# Example: wigner_d_order2_N5000.npz
#   5000 orientations
#   Uniformly sampled over SO(3)
#   Each matrix is 5×5 for rank-2 spherical harmonics
```

### Correlation Matrix Input
From `AutocorrelationCalculator.compute_correlation_matrix()`:

```python
corr_matrix = {
    (m1, m2): np.ndarray  # Shape: (n_lags,), dtype: complex128
    for m1 in [-2, -1, 0, 1, 2]
    for m2 in [-2, -1, 0, 1, 2]
}
# Total: 25 cross-correlations (5×5 matrix)
```

### Rotated Correlations Output
```python
rotated_corrs = np.ndarray  # Shape: (n_orientations, 5, 5, n_lags), dtype: complex128
# Dimensions:
#   axis 0: orientation index
#   axis 1: m1 index (m = -2,-1,0,1,2 → indices 0,1,2,3,4)
#   axis 2: m2 index
#   axis 3: lag index
```

### Ensemble Average Output
```python
ensemble_avg = np.ndarray  # Shape: (5, 5, n_lags), dtype: complex128
# Average over all orientations
# Used for next steps: spectral density → T1/T2 calculation
```

## Performance Benchmarks

### With Numba (Recommended)
```
Configuration: 5000 orientations, 2000 lags, 5×5 correlation matrix

Operation                    Time
--------------------------------------------------
Load Wigner library         ~0.1 s
Rotate correlation matrix   ~2-5 s
Compute ensemble average    ~0.01 s
Save ensemble average       ~0.05 s
--------------------------------------------------
Total                       ~2-5 s
```

### Without Numba (Fallback)
```
Configuration: 5000 orientations, 2000 lags, 5×5 correlation matrix

Operation                    Time
--------------------------------------------------
Load Wigner library         ~0.1 s
Rotate correlation matrix   ~30-60 s
Compute ensemble average    ~0.01 s
Save ensemble average       ~0.05 s
--------------------------------------------------
Total                       ~30-60 s
```

**Recommendation**: Install numba for 10-20× speedup:
```bash
conda install numba
# or
pip install numba
```

## Validation

Module 5 has been validated with 7 comprehensive tests:

1. ✅ Load Wigner D-matrix library
2. ✅ Compute Wigner D-matrices from Euler angles
3. ✅ Rotate correlation matrix
4. ✅ Ensemble averaging
5. ✅ Save and load functionality
6. ✅ `rotate_all()` function
7. ✅ Reference implementation comparison

Run tests:
```bash
cd analysis
python -m nmr_calculator.test_rotated_correlation
```

## Next Steps

### Module 6: Spectral Density
Use rotated correlations to calculate spectral density:

```python
from nmr_calculator import SpectralDensityCalculator

# Extract specific component
acf_m1 = rotated_corrs[:, 3, 3, :]  # m=1 component

# Calculate spectral density
sd_calc = SpectralDensityCalculator(config)
frequencies, J = sd_calc.calculate(acf_m1.T, apply_window=False)
```

### Module 7-9: T1/T2 Calculation
Use spectral densities to compute relaxation rates.

## References

1. Reference implementation: `t1_anisotropy_analysis.py`
   - Lines 356-383: `rotate_all()` function
   - Lines 517-570: `calculate_ensemble_T1()`
   - Lines 673-684: Main workflow

2. Theory:
   - Wigner rotation matrices: Varshalovich et al., "Quantum Theory of Angular Momentum"
   - NMR relaxation: Cavanagh et al., "Protein NMR Spectroscopy"

## API Reference

### RotatedCorrelationCalculator

```python
class RotatedCorrelationCalculator:
    """Calculate rotated correlation matrices for ensemble-averaged NMR."""
    
    def __init__(self, config: NMRConfig):
        """Initialize calculator."""
    
    def load_wigner_d_library(self, library_path: str) -> bool:
        """Load pre-computed Wigner D-matrix library."""
    
    def compute_wigner_d_from_euler(self, euler_angles: np.ndarray) -> np.ndarray:
        """Compute Wigner D-matrices from Euler angles."""
    
    def rotate_correlation_matrix(
        self, 
        correlation_matrix: Dict[Tuple[int, int], np.ndarray],
        save_individual: bool = False,
        save_dir: Optional[str] = None
    ) -> np.ndarray:
        """Rotate correlation matrix using Wigner D-matrices."""
    
    def compute_ensemble_average(
        self,
        rotated_corrs: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """Compute ensemble-averaged correlation matrix."""
    
    def load_ensemble_average(self, load_path: str) -> np.ndarray:
        """Load previously computed ensemble average."""
```

### Standalone Function

```python
def rotate_all(D_all: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    Rotate correlation matrix using Wigner D-matrices.
    
    Parameters
    ----------
    D_all : np.ndarray
        Wigner D-matrices (n_orientations, 5, 5)
    A : np.ndarray
        Correlation matrix (5, 5, n_lags)
    
    Returns
    -------
    rotated_corrs : np.ndarray
        Rotated correlation matrices (n_orientations, 5, 5, n_lags)
    """
```

## Files

- `rotated_correlation.py`: Main implementation
- `test_rotated_correlation.py`: Test suite (7 tests)
- `MODULE5_README.md`: This documentation

## Status

✅ **Module 5 Complete**
- All functionality implemented
- All tests passing (7/7)
- Matches reference implementation
- Ready for integration with spectral density module
