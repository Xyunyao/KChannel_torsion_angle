"""
NMR Calculator Package

A modular Python package for calculating NMR relaxation parameters from 
molecular dynamics trajectories or simulated molecular motions.

Modules:
--------
- config: Configuration system with NMRConfig dataclass
- xyz_generator: Trajectory generation (diffusion on cone, custom, file loading)
- euler_converter: XYZ to Euler angle conversion with local axis definitions
- spherical_harmonics: Y₂ₘ decomposition for CSA and dipolar interactions
- autocorrelation: Fast ACF calculation using FFT
- rotation_matrix: Wigner-D rotation matrices for spherical harmonics
- spectral_density: J(ω) calculation with zero-filling and frequency markers
- nmr_parameters: T1, T2, NOE calculations from spectral density
- nmr_pipeline: Complete orchestrator combining all modules

Usage:
------
    from nmr_calculator import NMRConfig, NMRPipeline
    
    config = NMRConfig(
        trajectory_type='diffusion_cone',
        S2=0.85,
        tau_c=5e-9,
        B0=14.1,
        nucleus='15N',
        interaction_type='CSA',
        delta_sigma=160.0,
        verbose=True
    )
    
    pipeline = NMRPipeline(config)
    results = pipeline.run()
    
    print(f"T1 = {results['T1']*1000:.1f} ms")
"""

__version__ = '1.0.0'

# Import main classes for easy access
from .config import NMRConfig, GAMMA, get_larmor_frequency
from .nmr_pipeline import NMRPipeline
from .xyz_generator import TrajectoryGenerator
from .euler_converter import EulerConverter
from .spherical_harmonics import SphericalHarmonicsCalculator
from .autocorrelation import AutocorrelationCalculator
from .rotation_matrix import WignerDCalculator
from .rotated_correlation import RotatedCorrelationCalculator, rotate_all
from .spectral_density import SpectralDensityCalculator
from .nmr_parameters import NMRParametersCalculator

__all__ = [
    # Main interface
    'NMRConfig',
    'NMRPipeline',
    
    # Individual modules
    'TrajectoryGenerator',
    'EulerConverter',
    'SphericalHarmonicsCalculator',
    'AutocorrelationCalculator',
    'WignerDCalculator',
    'RotatedCorrelationCalculator',
    'SpectralDensityCalculator',
    'NMRParametersCalculator',
    
    # Functions
    'rotate_all',
    
    # Constants and utilities
    'GAMMA',
    'get_larmor_frequency',
]
