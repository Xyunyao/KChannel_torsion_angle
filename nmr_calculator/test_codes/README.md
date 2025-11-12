# NMR Calculator Test Suite

This directory contains all test files for the NMR Calculator package.

## Directory Structure

```
analysis/
├── nmr_calculator/           # Main package
│   ├── config.py
│   ├── xyz_generator.py
│   ├── euler_converter.py
│   ├── spherical_harmonics.py
│   ├── autocorrelation.py
│   ├── spectral_density.py
│   ├── nmr_parameters.py
│   └── ...
└── test_codes/              # Test files (this directory)
    ├── test_*.py
    └── test_*.ipynb
```

## Running Tests

All test files have been moved to this directory and updated with correct import paths.

### Run from test_codes directory:

```bash
cd /Users/yunyao_1/Dropbox/KcsA/analysis/nmr_calculator/test_codes
python test_nmr_parameters.py
python test_csa_formulas_eta0.py
python test_lipari_szabo_acf_comparison.py
# ... etc
```

### Run from any directory:

```bash
python /path/to/analysis/nmr_calculator/test_codes/test_nmr_parameters.py
```

## Import Path Setup

All test files include the following path setup to import the nmr_calculator package:

### For files using `from nmr_calculator.module import ...`:
```python
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
```
This adds the `analysis/` directory to Python path (3 levels up from test file).

### For files using `from module import ...` (direct imports):
```python
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```
This adds the `nmr_calculator/` directory to Python path (2 levels up from test file).

## Test Files

### Correlation and Autocorrelation Tests
- `test_acf_methods.py` - ACF calculation methods comparison
- `test_autocorrelation_improvements.py` - ACF performance improvements
- `test_autocorrelation_updated.py` - Updated ACF implementation
- `test_correlation_matrix_dual.py` - Direct vs FFT correlation matrix
- `test_correlation_matrix_final.py` - Final correlation matrix validation
- `test_dual_acf_methods.py` - Dual ACF method comparison

### Lipari-Szabo Model Tests
- `test_lipari_szabo_simple.py` - Simple Lipari-Szabo comparison
- `test_lipari_szabo_validation.py` - Full Lipari-Szabo validation
- `test_lipari_szabo_acf_comparison.py` - ACF comparison
- `test_t1_lipari_szabo_vs_simulation.py` - T1 analytical vs simulated

### CSA Formula Tests
- `test_csa_formulas_eta0.py` - CSA T1 formulas for η=0
- `test_csa_formulas_eta0.ipynb` - Jupyter notebook version
- `test_csa_t1_comparison.py` - CSA T1 comparison

### Spectral Density Tests
- `test_fft_no_dc.py` - FFT with DC offset removal
- `test_spherical_harmonics_full.py` - Spherical harmonics validation

### NMR Parameters Tests
- `test_nmr_parameters.py` - T1, T2, NOE calculations

### Rotation Matrix Tests
- `test_rotated_correlation.py` - Rotated correlation functions
- `test_rotated_correlation_analytical.py` - Analytical comparison

## Notes

1. All test files maintain their original functionality
2. Import paths have been updated to work from the test_codes directory
3. Test files can still be run individually
4. Some tests may have API-related issues due to code evolution (not path-related)

## Recent Updates

- **Date**: November 11, 2025
- **Change**: Moved all test files from `nmr_calculator/` to `nmr_calculator/test_codes/`
- **Path fixes**: Updated all `sys.path.insert()` statements to use correct relative paths
- **Files moved**: 17 Python test files + 1 Jupyter notebook
