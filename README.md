# NMR Parameter Calculator

A modular Python package for calculating NMR relaxation parameters from molecular dynamics trajectories.

## 📦 Package Structure

```
nmr_calculator/
├── __init__.py                 # Package initialization
├── config.py                   # Configuration and constants ✅
├── xyz_generator.py            # Module 1: Trajectory generation ✅
├── euler_converter.py          # Module 2: XYZ → Euler angles
├── spherical_harmonics.py      # Module 3: Interaction → Y series
├── autocorrelation.py          # Module 4: ACF calculation
├── rotation_matrix.py          # Module 5: Wigner-D rotations
├── spectral_density.py         # Module 6: J(ω) calculation
├── nmr_parameters.py           # Module 7: T1, T2 calculation
├── nmr_pipeline.py            # Global orchestrator
└── utils.py                    # Utility functions
```

## 🎯 Modules Overview

### ✅ Module 0: Configuration (`config.py`)
**Status: COMPLETE**

- Physical constants (ℏ, γ for H/C/N/P)
- `NMRConfig` dataclass with all parameters
- Larmor frequency calculations
- Configuration summary printing

### ✅ Module 1: XYZ Generator (`xyz_generator.py`)
**Status: COMPLETE**

**Features:**
- **Diffusion on cone** (Lipari-Szabo model) - default
  - Input: S², τc, dt, num_steps
  - Output: List of rotation matrices
- **Custom trajectory** with placeholder for user function
- **Load from file** (.xyz, .npz) with placeholder

**Usage:**
```python
from nmr_calculator import NMRConfig
from nmr_calculator.xyz_generator import TrajectoryGenerator

config = NMRConfig(trajectory_type='diffusion_cone', S2=0.85, tau_c=2e-9)
generator = TrajectoryGenerator(config)
rotations, coords = generator.generate()
```

### 🔄 Module 2: Euler Converter (`euler_converter.py`)
**Status: IN PROGRESS**

**Will include:**
- CO-Cα local axis definition (default for backbone)
- N-H local axis definition
- Custom axis definition (placeholder)
- XYZ coordinates → Euler angles (ZYZ convention)

### 🔄 Module 3: Spherical Harmonics (`spherical_harmonics.py`)
**Status: IN PROGRESS**

**Will include:**
- CSA interaction → rank-2 spherical harmonics
- Dipolar interaction → rank-2 spherical harmonics
- Quadrupolar interaction (placeholder)
- Wigner rotation from Euler angles

### 🔄 Module 4: Autocorrelation (`autocorrelation.py`)
**Status: IN PROGRESS**

**Will include:**
- Fast autocorrelation via FFT
- Options: max_lag, lag_step
- Normalization (C(0) = 1)
- Progress tracking for long calculations

### 🔄 Module 5: Rotation Matrix (`rotation_matrix.py`)
**Status: IN PROGRESS**

**Will include:**
- Load Wigner-D library
- Apply rotations to ACF
- Ensemble-averaged ACF
- Individual rotation matrices save option
- Output: (Euler angles, rotated correlation matrix)

### 🔄 Module 6: Spectral Density (`spectral_density.py`)
**Status: IN PROGRESS**

**Will include:**
- FFT-based J(ω) calculation
- Zero-filling option (factor = 1, 2, 4, 8, ...)
- Frequency markers for H/C/N/P
- Smoothing (Savitzky-Golay)
- Visualization with marked frequencies

### 🔄 Module 7: NMR Parameters (`nmr_parameters.py`)
**Status: IN PROGRESS**

**Will include:**
- T1 calculation from J(ω₀)
- T2 calculation (placeholder)
- Multiple nucleus support
- Field-dependent calculations

### 🔄 Module 8: Pipeline (`nmr_pipeline.py`)
**Status: IN PROGRESS**

**Will include:**
- Global orchestrator combining all modules
- Step-by-step execution with intermediate saves
- Resume from saved intermediate results
- Comprehensive logging and progress tracking

## 🚀 Quick Start

```python
from nmr_calculator import NMRPipeline, NMRConfig

# Configure calculation
config = NMRConfig(
    trajectory_type='diffusion_cone',
    S2=0.85,
    tau_c=2e-9,
    B0=14.1,  # 14.1 T (600 MHz for 1H)
    nucleus='13C',
    interaction_type='CSA',
    delta_sigma=50.0,  # ppm
    calculate_T1=True,
    output_dir='./results',
    save_intermediate=True
)

# Run pipeline
pipeline = NMRPipeline(config)
results = pipeline.run()

# Access results
print(f"T1 = {results['T1']:.4f} s")
```

## 💡 Key Features

### Modular Design
Each module can be used independently or as part of the pipeline

### Flexible Configuration
All parameters controlled through `NMRConfig` dataclass

### Intermediate Saves
Save results at any step for inspection or restart

### Extensible
Easy to add custom trajectory generators, axis definitions, or interactions

### Well-Documented
Comprehensive docstrings and type hints

## 🔧 Installation

```bash
cd /Users/yunyao_1/Dropbox/KcsA/analysis
# The package is ready to use in-place, or install in development mode:
pip install -e nmr_calculator/
```

## 📊 Workflow

```
1. Generate/Load Trajectory
   ↓ (rotations or coordinates)
2. Convert to Euler Angles
   ↓ (Euler angles in lab frame)
3. Calculate Spherical Harmonics
   ↓ (Y₂ₘ time series)
4. Compute Autocorrelation
   ↓ (C(t) for each m)
5. Apply Wigner-D Rotations
   ↓ (Rotated ACF)
6. Calculate Spectral Density
   ↓ (J(ω) via FFT)
7. Calculate NMR Parameters
   ↓ (T1, T2, etc.)
```

## 📝 TODO

- [ ] Complete Module 2 (euler_converter.py)
- [ ] Complete Module 3 (spherical_harmonics.py)
- [ ] Complete Module 4 (autocorrelation.py)
- [ ] Complete Module 5 (rotation_matrix.py)
- [ ] Complete Module 6 (spectral_density.py)
- [ ] Complete Module 7 (nmr_parameters.py)
- [ ] Complete Module 8 (nmr_pipeline.py)
- [ ] Add unit tests for each module
- [ ] Add example scripts
- [ ] Add validation against analytical solutions

## 🤝 Contributing

This is a research package. Suggestions for improvements:

1. **Performance**: Use numba/cython for hot loops
2. **File formats**: Support more trajectory formats (DCD, TRR, XTC)
3. **Interactions**: Add dipole-dipole, quadrupolar
4. **Analysis**: Add correlation time fitting, model-free analysis
5. **Visualization**: Enhanced plotting utilities

## 📚 References

1. Lipari & Szabo (1982) - Model-free approach
2. Abragam (1961) - Principles of Nuclear Magnetism
3. Palmer et al. (2001) - NMR relaxation review

## ✅ Completed

- ✅ Package structure
- ✅ Configuration module with all constants
- ✅ Module 1: Trajectory generation (diffusion on cone)
- ✅ Modular design with independent modules
- ✅ Comprehensive documentation

## 🎯 Current Status

**Modules Complete: 2/9**
- ✅ config.py
- ✅ xyz_generator.py
- 🔄 euler_converter.py (next)
- ⏸️ spherical_harmonics.py
- ⏸️ autocorrelation.py
- ⏸️ rotation_matrix.py
- ⏸️ spectral_density.py
- ⏸️ nmr_parameters.py
- ⏸️ nmr_pipeline.py

Would you like me to continue with the remaining modules?
