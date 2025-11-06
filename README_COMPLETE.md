# NMR Calculator Package

A comprehensive, modular Python package for calculating NMR relaxation parameters (T1, T2, NOE) from molecular dynamics trajectories or simulated molecular motions.

## Features

- **Modular Design**: Each calculation step is an independent module
- **Flexible Input**: Generate trajectories or load from MD simulations
- **Multiple Interactions**: CSA (chemical shift anisotropy) and dipolar coupling
- **Efficient Algorithms**: FFT-based spectral density with zero-padding
- **Comprehensive Output**: Save intermediate results at each step
- **Well-Documented**: Detailed docstrings and examples

## Module Structure

1. **config.py** - Configuration management
2. **xyz_generator.py** - Trajectory generation (full rotation matrices)
3. **cone_simulation.py** - Vector cone diffusion utilities (for testing/validation)
4. **spectral_density.py** - Spectral density calculations
5. **csa_tensor.py** - CSA tensor operations
6. **dipolar_coupling.py** - Dipolar coupling calculations
7. **relaxation.py** - T1, T2, NOE calculations
8. **ensemble_averaging.py** - Ensemble averaging over orientations
9. **output_writer.py** - Results export (JSON, CSV, plots)
10. **main_pipeline.py** - Orchestration of all modules

## Installation

```bash
# Navigate to analysis directory
cd /Users/yunyao_1/Dropbox/KcsA/analysis

# The package is ready to use (no installation needed)
# Just ensure you have dependencies:
pip install numpy scipy
```

## Quick Start

### Example 1: Basic T1 Calculation (Within Cone)

```python
from nmr_calculator import NMRConfig, NMRPipeline

# Configure
config = NMRConfig(
    # Trajectory - diffusion within cone volume
    trajectory_type='diffusion_cone',
    S2=0.85,           # Order parameter
    tau_c=5e-9,        # Correlation time (5 ns)
    dt=1e-12,          # Time step (1 ps)
    num_steps=20000,
    
    # NMR parameters
    B0=14.1,           # Magnetic field (Tesla)
    nucleus='15N',     # Observed nucleus
    interaction_type='CSA',
    delta_sigma=160.0, # CSA (ppm)
    
    # Options
    calculate_T1=True,
    verbose=True
)

# Run pipeline
pipeline = NMRPipeline(config)
results = pipeline.run()

# Results
print(f"T1 = {results['T1']*1000:.1f} ms")
```

### Example 1b: T1 with Cone Edge Model

```python
# Alternative: Diffusion on cone surface (edge)
config = NMRConfig(
    trajectory_type='diffusion_cone_edge',  # Fixed β at cone angle
    S2=0.85,
    tau_c=5e-9,
    dt=1e-12,
    num_steps=20000,
    B0=14.1,
    nucleus='15N',
    interaction_type='CSA',
    delta_sigma=160.0,
    calculate_T1=True,
    verbose=True
)

pipeline = NMRPipeline(config)
results = pipeline.run()
print(f"T1 = {results['T1']*1000:.1f} ms")
```

### Example 2: Dipolar Relaxation with NOE

```python
config = NMRConfig(
    trajectory_type='diffusion_cone',
    S2=0.85,
    tau_c=5e-9,
    dt=1e-12,
    num_steps=20000,
    B0=14.1,
    nucleus='15N',
    interaction_type='dipolar',  # 15N-1H dipolar
    calculate_T1=True,
    calculate_T2=True,
    verbose=True
)

pipeline = NMRPipeline(config)
results = pipeline.run()

print(f"T1  = {results['T1']*1000:.1f} ms")
print(f"T2  = {results['T2']*1000:.1f} ms")
print(f"NOE = {results['NOE']:.3f}")
```

### Example 3: Use Individual Modules

```python
from nmr_calculator import (
    NMRConfig,
    TrajectoryGenerator,
    EulerConverter,
    SphericalHarmonicsCalculator,
    AutocorrelationCalculator,
    SpectralDensityCalculator,
    NMRParametersCalculator
)

# Configure
config = NMRConfig(
    trajectory_type='diffusion_cone',
    S2=0.85,
    tau_c=5e-9,
    B0=14.1,
    nucleus='15N',
    interaction_type='CSA',
    delta_sigma=160.0
)

# Step by step
gen = TrajectoryGenerator(config)
rotations, coords = gen.generate()

converter = EulerConverter(config)
euler_angles = converter.convert(rotations=rotations)

sh_calc = SphericalHarmonicsCalculator(config)
Y2m = sh_calc.calculate(euler_angles)

acf_calc = AutocorrelationCalculator(config)
acf, time_lags = acf_calc.calculate(Y2m)

sd_calc = SpectralDensityCalculator(config)
J, frequencies = sd_calc.calculate(acf, time_lags)

nmr_calc = NMRParametersCalculator(config)
T1, T2 = nmr_calc.calculate(J, frequencies)

print(f"T1 = {T1*1000:.1f} ms")
```

## Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                    NMR CALCULATION PIPELINE                 │
└─────────────────────────────────────────────────────────────┘

1. xyz_generator        →  Generate/load trajectory
   ├─ Diffusion WITHIN cone (Lipari-Szabo)
   │    β varies: 0 ≤ β ≤ θ_cone
   │    S² = ((1 + cos(θ))/2)²
   │
   ├─ Diffusion ON cone edge/surface
   │    β fixed: β = θ_cone (constant)
   │    Azimuthal diffusion only
   │    S² = ((1 + cos(θ)) × cos(θ) / 2)²
   │
   ├─ Custom trajectories
   └─ Load from files (.xyz, .npz)
                        ↓
2. euler_converter      →  XYZ → Euler angles (α, β, γ)
   ├─ CO-Cα axis (backbone carbonyl)
   ├─ N-H axis (amide)
   └─ Custom axis definitions
                        ↓
3. spherical_harmonics  →  Euler → Y₂ₘ coefficients
   ├─ CSA tensor decomposition
   └─ Dipolar coupling (axial)
                        ↓
4. autocorrelation      →  C(τ) = ⟨Y₂ₘ(t)·Y₂ₘ*(t+τ)⟩
   ├─ FFT-based calculation
   ├─ Lag options
   └─ Exponential fitting
                        ↓
5. spectral_density     →  J(ω) = FFT{C(τ)}
   ├─ Zero-padding for resolution
   ├─ Frequency markers (ωN, ωH, etc.)
   └─ Analytical comparison
                        ↓
6. nmr_parameters       →  T1, T2, NOE
   ├─ CSA relaxation
   ├─ Dipolar relaxation
   └─ Multiple field strengths

        ↓
    RESULTS: T1, T2, NOE, R1, R2
```

## Configuration Parameters

### Trajectory Generation
- `trajectory_type`: 
  - `'diffusion_cone'` - Diffusion within cone volume (default)
  - `'diffusion_cone_edge'` - Diffusion on cone surface only
  - `'custom'` - User-defined trajectory
  - `'from_file'` - Load from MD trajectory
- `S2`: Order parameter (0 to 1)
- `tau_c`: Correlation time (seconds)
- `dt`: Time step (seconds)
- `num_steps`: Number of trajectory points

### NMR Parameters
- `B0`: Magnetic field strength (Tesla)
- `nucleus`: '1H', '13C', '15N', '31P'
- `interaction_type`: 'CSA' or 'dipolar'
- `delta_sigma`: CSA anisotropy (ppm)
- `eta`: CSA asymmetry (0 to 1)

### Local Axis Definition
- `local_axis_definition`: 'CO_CA', 'NH', 'custom'

### Calculation Options
- `max_lag`: Maximum lag for ACF
- `lag_step`: Lag step size
- `zero_fill_factor`: Zero-padding multiplier
- `frequency_markers`: Calculate J at specific ω

### Output Options
- `calculate_T1`: Calculate longitudinal relaxation
- `calculate_T2`: Calculate transverse relaxation
- `output_dir`: Directory for results
- `save_intermediate`: Save each step
- `verbose`: Print progress

## Physical Constants

Gyromagnetic ratios (γ) in rad/(T·s):
- ¹H: 2π × 42.577 MHz/T
- ¹³C: 2π × 10.705 MHz/T
- ¹⁵N: 2π × -4.316 MHz/T (negative!)
- ³¹P: 2π × 17.235 MHz/T

## Theory

### Cone Diffusion Models

**Model 1: Diffusion Within Cone (volume)**
```
Motion: 0 ≤ β ≤ θ_cone (variable polar angle)
S² = ((1 + cos(θ_cone))/2)²
Explores full cone interior
```

**Model 2: Diffusion On Cone Edge (surface)**
```
Motion: β = θ_cone (fixed polar angle)
       α = free diffusion (azimuthal)
S² = ((1 + cos(θ_cone)) × cos(θ_cone) / 2)²
Restricted to cone surface
For same S², θ_edge > θ_within
```

### Lipari-Szabo Model
```
C(τ) = S² + (1 - S²)·exp(-τ/τe)
J(ω) = (2/5)·[S²τc/(1+(ωτc)²) + (1-S²)τe/(1+(ωτe)²)]
```

### CSA Relaxation
```
R1_CSA = (2/15)·(ωN·Δσ)²·[J(ωN) + 3J(ωN) + 6J(2ωN)]
R2_CSA = (1/15)·(ωN·Δσ)²·[4J(0) + 3J(ωN) + 6J(2ωN)]
```

### Dipolar Relaxation (¹⁵N-¹H)
```
d = (μ₀/4π)·(γH·γN·ℏ)/r³NH
R1_DD = (d²/4)·[J(ωH-ωN) + 3J(ωN) + 6J(ωH+ωN)]
R2_DD = (d²/8)·[4J(0) + J(ωH-ωN) + 3J(ωN) + 6J(ωH) + 6J(ωH+ωN)]
NOE = 1 + (γH/γN)·(d²/4R1)·[6J(ωH+ωN) - J(ωH-ωN)]
```

## Examples

See `example_usage.py` for complete examples:

1. **Basic CSA T1 calculation**
2. **Dipolar relaxation with NOE**
3. **Plot spectral density**
4. **Plot autocorrelation function**
5. **Parameter scan** (T1 vs τc)
6. **Compare CSA vs dipolar**

See `example_cone_comparison.py` for cone model comparison:
- Visual comparison of within-cone vs edge diffusion
- 3D cone geometry visualization
- Angle distributions and time series
- S² validation from trajectories

See `cone_simulation.py` for vector-based cone simulation utilities:
- `simulate_vector_on_cone()` - Diffusion on cone edge (fixed θ)
- `simulate_vector_within_cone()` - Diffusion within cone volume
- `calculate_order_parameter()` - Calculate S² from vectors
- Useful for testing and generating synthetic data

Run examples:
```bash
cd nmr_calculator
python example_usage.py
python example_cone_comparison.py
python cone_simulation.py  # Test vector simulations
```

## Output Files

When `save_intermediate=True`, the pipeline saves:

- `step1_trajectory.npz`: Rotation quaternions and coordinates
- `step2_euler_angles.npz`: Euler angles (ZYZ convention)
- `step3_Y2m_coefficients.npz`: Spherical harmonics coefficients
- `step4_autocorrelation.npz`: ACF and time lags
- `step5_spectral_density.npz`: J(ω) and frequencies
- `step6_nmr_parameters.npz`: T1, T2, NOE

## Customization

### Add Custom Trajectory

```python
def my_custom_trajectory(config):
    """Generate custom molecular motion."""
    rotations = []
    for i in range(config.num_steps):
        # Your custom rotation logic
        angle = ...
        rotation = R.from_euler('ZYZ', [angle, 0, 0])
        rotations.append(rotation)
    return rotations

config = NMRConfig(trajectory_type='custom')
config.custom_trajectory_function = my_custom_trajectory
```

### Add Custom Local Axis

```python
def my_axis_extraction(coordinates):
    """Extract local z-axis from coordinates."""
    z_axis = ...  # Your logic
    return z_axis

config = NMRConfig(local_axis_definition='custom')
config.custom_axis_vectors = my_axis_extraction
```

## Testing

Validate against analytical Lipari-Szabo:

```python
from nmr_calculator import SpectralDensityCalculator

# After calculating spectral density
sd_calc = SpectralDensityCalculator(config)
J, freq = sd_calc.calculate(acf, time_lags)

# Compare with analytical
rmse, max_error = sd_calc.compare_with_analytical()
print(f"RMSE: {rmse:.2e}")
print(f"Max error: {max_error:.2%}")
```

## Dependencies

- NumPy >= 1.20
- SciPy >= 1.7

Optional:
- Matplotlib (for plotting examples)

## References

1. Lipari & Szabo (1982). "Model-free approach to the interpretation of nuclear magnetic resonance relaxation in macromolecules."
2. Palmer (2004). "NMR characterization of the dynamics of biomacromolecules."
3. Cavanagh et al. (2007). "Protein NMR Spectroscopy" (textbook)

## Author

Built for KcsA NMR analysis project

## License

Internal research use

---

**Status**: ✅ All 9 modules complete and functional  
**Version**: 1.0.0  
**Last Updated**: 2024
