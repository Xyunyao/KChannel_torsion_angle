"""
Configuration module for NMR parameter calculations.

Defines physical constants, default parameters, and nucleus-specific properties.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional


# Physical constants
PLANCK_H = 6.62607015e-34  # J·s
HBAR = PLANCK_H / (2 * np.pi)  # J·s

# Gyromagnetic ratios (rad/(s·T))
GAMMA = {
    '1H': 2 * np.pi * 42.577e6,
    '13C': 2 * np.pi * 10.705e6,
    '15N': 2 * np.pi * -4.316e6,
    '31P': 2 * np.pi * 17.235e6,
    '2H': 2 * np.pi * 6.536e6,
    '19F': 2 * np.pi * 40.078e6,
}


def get_larmor_frequency(nucleus: str, B0: float) -> float:
    """
    Calculate Larmor frequency for a nucleus at given field strength.
    
    Parameters
    ----------
    nucleus : str
        Nucleus identifier (e.g., '1H', '13C', '15N', '31P')
    B0 : float
        Magnetic field strength in Tesla
    
    Returns
    -------
    float
        Larmor frequency in Hz
    """
    if nucleus not in GAMMA:
        raise ValueError(f"Unknown nucleus: {nucleus}. Available: {list(GAMMA.keys())}")
    
    omega0 = GAMMA[nucleus] * B0  # rad/s
    return omega0 / (2 * np.pi)  # Hz


@dataclass
class NMRConfig:
    """
    Configuration class for NMR parameter calculations.
    
    Attributes
    ----------
    # Magnetic field
    B0 : float
        Magnetic field strength in Tesla (default: 14.1 T for 600 MHz 1H)
    
    # Nucleus selection
    nucleus : str
        Target nucleus for NMR calculations (default: '13C')
    
    # Trajectory generation (Module 1)
    trajectory_type : str
        Type of trajectory: 'diffusion_cone', 'custom', 'from_file'
    S2 : float
        Order parameter for diffusion on cone (0 < S2 < 1)
    tau_c : float
        Correlation time in seconds
    dt : float
        Time step in seconds
    num_steps : int
        Number of simulation steps
    
    # Euler angle conversion (Module 2)
    local_axis_definition : str
        Local axis definition: 'CO_CA' (default), 'NH', 'custom'
    custom_axis_vectors : Optional[Tuple]
        Custom axis vectors if local_axis_definition='custom'
    
    # Interaction type (Module 3)
    interaction_type : str
        Type of interaction: 'CSA', 'dipolar', 'quadrupolar'
    csa_tensor : Optional[np.ndarray]
        CSA tensor in principal axis frame (3x3)
    delta_sigma : Optional[float]
        CSA anisotropy in ppm
    eta : float
        CSA asymmetry parameter (0 <= eta <= 1)
    D_coupling : Optional[float]
        Dipolar coupling constant in Hz (e.g., 10000 Hz for 15N-1H at 1.02 Å)
    
    # Autocorrelation (Module 4)
    max_lag : int
        Maximum lag for autocorrelation calculation
    lag_step : int
        Step size for lag sampling
    
    # Rotation matrix (Module 5)
    wigner_d_library : Optional[str]
        Path to Wigner-D matrix library file
    save_ensemble_averaged : bool
        Save ensemble-averaged correlation matrix
    save_individual_rotations : bool
        Save individual rotation matrices
    
    # Spectral density (Module 6)
    zero_fill_factor : int
        Zero-padding factor for FFT (1 = no padding)
    smoothing_window : int
        Window size for moving average smoothing of spectral density (1 = no smoothing)
    frequency_markers : list
        List of nuclei to mark on spectral density plot
    
    # NMR parameters (Module 7)
    calculate_T1 : bool
        Calculate T1 relaxation time
    calculate_T2 : bool
        Calculate T2 relaxation time (placeholder)
    
    # Output options
    output_dir : str
        Directory for output files
    save_intermediate : bool
        Save intermediate results at each step
    verbose : bool
        Print detailed progress information
    """
    
    # Magnetic field
    B0: float = 14.1  # Tesla (600 MHz for 1H)
    
    # Nucleus
    nucleus: str = '13C'
    
    # Trajectory generation
    trajectory_type: str = 'diffusion_cone'
    S2: float = 0.85
    tau_c: float = 2e-9  # 2 ns
    dt: float = 2e-12  # 2 ps
    num_steps: int = 20000
    
    # Euler angle conversion
    local_axis_definition: str = 'CO_CA'
    custom_axis_vectors: Optional[Tuple] = None
    
    # Interaction type
    interaction_type: str = 'CSA'
    csa_tensor: Optional[np.ndarray] = None
    delta_sigma: float = 50.0  # ppm
    eta: float = 0.0  # Axially symmetric
    D_coupling: Optional[float] = None  # Dipolar coupling constant in Hz (e.g., 10000 Hz for 15N-1H)
    
    # Autocorrelation
    max_lag: int = 5000
    lag_step: int = 1
    
    # Rotation matrix
    wigner_d_library: Optional[str] = None
    save_ensemble_averaged: bool = True
    save_individual_rotations: bool = False
    
    # Spectral density
    zero_fill_factor: int = 1  # No zero-filling by default
    smoothing_window: int = 5  # Moving average window size for spectral density smoothing
    frequency_markers: list = field(default_factory=lambda: ['1H', '13C', '15N', '31P'])
    
    # NMR parameters
    calculate_T1: bool = True
    calculate_T2: bool = False
    
    # Output
    output_dir: str = './nmr_results'
    save_intermediate: bool = True
    verbose: bool = True
    
    def get_larmor_frequency(self) -> float:
        """Get Larmor frequency for configured nucleus and field."""
        return get_larmor_frequency(self.nucleus, self.B0)
    
    def get_omega0(self) -> float:
        """Get Larmor frequency in rad/s."""
        return GAMMA[self.nucleus] * self.B0
    
    def get_marker_frequencies(self) -> Dict[str, float]:
        """Get Larmor frequencies for all marker nuclei."""
        return {nuc: get_larmor_frequency(nuc, self.B0) 
                for nuc in self.frequency_markers if nuc in GAMMA}
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary for saving."""
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                config_dict[key] = value.tolist()
            else:
                config_dict[key] = value
        return config_dict
    
    def summary(self) -> str:
        """Generate human-readable summary of configuration."""
        omega0 = self.get_omega0()
        f0 = omega0 / (2 * np.pi)
        
        summary = f"""
╔══════════════════════════════════════════════════════════════╗
║           NMR PARAMETER CALCULATION CONFIGURATION            ║
╚══════════════════════════════════════════════════════════════╝

Magnetic Field & Nucleus:
  • B₀ = {self.B0:.2f} T
  • Nucleus = {self.nucleus}
  • Larmor frequency = {f0*1e-6:.2f} MHz (ω₀ = {omega0:.3e} rad/s)

Trajectory:
  • Type = {self.trajectory_type}
  • Duration = {self.num_steps * self.dt * 1e9:.2f} ns
  • Timestep = {self.dt*1e12:.2f} ps
  • Steps = {self.num_steps}
"""
        
        if self.trajectory_type == 'diffusion_cone':
            summary += f"  • S² = {self.S2:.4f}\n"
            summary += f"  • τc = {self.tau_c*1e9:.2f} ns\n"
        
        summary += f"""
Interaction:
  • Type = {self.interaction_type}
"""
        if self.interaction_type == 'CSA':
            summary += f"  • Δσ = {self.delta_sigma:.1f} ppm\n"
            summary += f"  • η = {self.eta:.2f}\n"
        
        summary += f"""
Autocorrelation:
  • Max lag = {self.max_lag} steps ({self.max_lag * self.dt * 1e9:.2f} ns)
  • Lag step = {self.lag_step}

Spectral Density:
  • Zero-fill factor = {self.zero_fill_factor}×
  • Smoothing window = {self.smoothing_window}
  • Frequency markers = {', '.join(self.frequency_markers)}

NMR Parameters:
  • Calculate T1 = {self.calculate_T1}
  • Calculate T2 = {self.calculate_T2}

Output:
  • Directory = {self.output_dir}
  • Save intermediate results = {self.save_intermediate}
"""
        return summary
