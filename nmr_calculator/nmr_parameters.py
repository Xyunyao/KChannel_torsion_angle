"""
Module 7: NMR Relaxation Parameters

Calculate T1 and T2 relaxation times from spectral density.

For CSA relaxation:
R1 = (1/T1) = (2/15) × (γB₀Δσ)² × [J(ωN) + 3J(ωN) + 6J(2ωN)]

For dipolar relaxation (e.g., 15N-1H):
R1 = (1/T1) = (d²/4) × [J(ωH-ωN) + 3J(ωN) + 6J(ωH+ωN)]
where d = (μ₀/4π) × (γHγNℏ/r³NH)

T2 calculations include additional J(0) terms.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from .config import NMRConfig, GAMMA


class NMRParametersCalculator:
    """
    Calculate NMR relaxation parameters (T1, T2, NOE, etc.).
    
    Attributes
    ----------
    config : NMRConfig
        Configuration object
    T1 : float
        Longitudinal relaxation time (seconds)
    T2 : float
        Transverse relaxation time (seconds)
    NOE : float
        Nuclear Overhauser Effect
    R1_csa : float
        CSA contribution to R1
    R1_dipolar : float
        Dipolar contribution to R1
    """
    
    def __init__(self, config: NMRConfig):
        """
        Initialize NMR parameters calculator.
        
        Parameters
        ----------
        config : NMRConfig
            Configuration object
        """
        self.config = config
        self.T1 = None
        self.T2 = None
        self.NOE = None
        self.R1_csa = None
        self.R1_dipolar = None
        self.R2_csa = None
        self.R2_dipolar = None
        
    def calculate(self, 
                 spectral_density: np.ndarray,
                 frequencies: np.ndarray,
                 frequency_markers: Optional[Dict] = None) -> Tuple[float, Optional[float]]:
        """
        Calculate T1 and T2 from spectral density.
        
        Parameters
        ----------
        spectral_density : np.ndarray
            Spectral density J(ω) (n_freq,)
        frequencies : np.ndarray
            Angular frequencies (rad/s) (n_freq,)
        frequency_markers : Dict, optional
            Pre-calculated J(ω) at specific frequencies
        
        Returns
        -------
        T1 : float
            Longitudinal relaxation time (seconds)
        T2 : float or None
            Transverse relaxation time (seconds)
            None if calculate_T2=False in config
        """
        if self.config.verbose:
            print(f"\n{'='*70}")
            print("MODULE 7: Calculating NMR Relaxation Parameters")
            print(f"{'='*70}")
            print(f"  Nucleus: {self.config.nucleus}")
            print(f"  B₀: {self.config.B0} T")
            print(f"  Interaction type: {self.config.interaction_type}")
        
        # Get spectral density at required frequencies
        J_values = self._get_J_at_frequencies(spectral_density, frequencies, frequency_markers)
        
        # Calculate T1
        if self.config.calculate_T1:
            if self.config.interaction_type == 'CSA':
                # Use universal formula that works for any η
                # This expects correlation function in ppm² units
                T1 = self._calculate_T1_CSA_universal(J_values)
            elif self.config.interaction_type == 'dipolar':
                T1 = self._calculate_T1_dipolar(J_values)
            else:
                raise ValueError(f"Unknown interaction type: {self.config.interaction_type}")
            
            self.T1 = T1
            
            if self.config.verbose:
                print(f"  ✓ T1 = {T1:.3f} s ({T1*1000:.1f} ms)")
        
        # Calculate T2
        T2 = None
        if self.config.calculate_T2:
            if self.config.interaction_type == 'CSA':
                T2 = self._calculate_T2_CSA(J_values)
            elif self.config.interaction_type == 'dipolar':
                T2 = self._calculate_T2_dipolar(J_values)
            
            self.T2 = T2
            
            if self.config.verbose:
                print(f"  ✓ T2 = {T2:.3f} s ({T2*1000:.1f} ms)")
        
        return T1, T2
    
    def _get_J_at_frequencies(self,
                             spectral_density: np.ndarray,
                             frequencies: np.ndarray,
                             frequency_markers: Optional[Dict] = None) -> Dict[str, float]:
        """
        Extract J(ω) values at specific frequencies needed for relaxation.
        
        Required frequencies:
        - J(0)
        - J(ωN) or J(ωX) for heteronucleus
        - J(ωH) for proton
        - J(ωH±ωN) for sum/difference frequencies
        
        Parameters
        ----------
        spectral_density : np.ndarray
            Spectral density values
        frequencies : np.ndarray
            Frequency values (rad/s)
        frequency_markers : Dict, optional
            Pre-calculated frequency markers
        
        Returns
        -------
        J_values : Dict[str, float]
            J(ω) at required frequencies
        """
        J_values = {}
        
        # Get Larmor frequencies
        omega_nucleus = self.config.get_omega0()  # Observed nucleus (e.g., 15N)
        omega_H = 2 * np.pi * self.config.B0 * GAMMA['1H']  # Proton
        
        # J(0)
        J_values['J_0'] = spectral_density[0]
        
        # J(ω₀) for observed nucleus
        if frequency_markers and 'omega_nucleus' in frequency_markers:
            J_values['J_omega'] = frequency_markers['omega_nucleus'][1]
        else:
            idx = np.argmin(np.abs(frequencies - omega_nucleus))
            J_values['J_omega'] = spectral_density[idx]
        
        # For dipolar, need J(ωH), J(ωH±ωN)
        if self.config.interaction_type == 'dipolar':
            # J(ωH)
            idx_H = np.argmin(np.abs(frequencies - omega_H))
            J_values['J_omega_H'] = spectral_density[idx_H]
            
            # J(ωH - ωX) where X is the observed nucleus
            omega_diff = omega_H - omega_nucleus
            idx_diff = np.argmin(np.abs(frequencies - omega_diff))
            J_values['J_omega_H_minus_X'] = spectral_density[idx_diff]
            
            # J(ωH + ωX)
            omega_sum = omega_H + omega_nucleus
            idx_sum = np.argmin(np.abs(frequencies - omega_sum))
            J_values['J_omega_H_plus_X'] = spectral_density[idx_sum]
        
        if self.config.verbose:
            print(f"\n  Spectral density values:")
            for key, val in J_values.items():
                print(f"    {key}: {val:.2e}")
        
        return J_values
    
    def _calculate_T1_CSA_universal(self, J_values: Dict[str, float]) -> float:
        """
        Calculate T1 for CSA relaxation using universal formula.
        
        R1 = (1/T1) = f₀² × J(ω₀) × 10⁻¹²
        
        where f₀ is the Larmor frequency in Hz (NOT rad/s).
        
        This formula works for ANY CSA tensor (any η value) when the correlation 
        function is calculated from Y2m spherical harmonics with CSA in ppm.
        The 10⁻¹² factor accounts for ppm² units in the correlation function.
        
        This is the approach used in t1_anisotropy_analysis.py where:
        1. Calculate Y2m(t) with CSA tensor in ppm
        2. Compute rotated correlation matrix C(m,m',τ)
        3. Extract C(1,1,τ) (or other diagonal elements)
        4. Calculate spectral density J(ω)
        5. Use R1 = f₀² × J(ω₀) × 10⁻¹² (f₀ in Hz)
        
        IMPORTANT: This formula expects J(ω) from correlation functions that
        ALREADY CONTAIN the CSA magnitude in ppm. Do NOT use this with normalized
        correlation functions - use _calculate_T1_CSA_uniaxial() instead for that case.
        
        Parameters
        ----------
        J_values : Dict[str, float]
            Spectral density at required frequencies (from ppm² correlation)
        
        Returns
        -------
        T1 : float
            Longitudinal relaxation time (seconds)
        """
        omega_0 = self.config.get_omega0()  # rad/s
        larmor_frequency_Hz = omega_0 / (2 * np.pi)  # Hz
        
        # J(ω₀) - spectral density at Larmor frequency
        J_omega_0 = J_values['J_omega']
        
        # R1 = f₀² × J(ω₀) × 10⁻¹²
        # where f₀ is in Hz (NOT rad/s!)
        # The 10⁻¹² comes from ppm² in the correlation function
        # This matches the reference implementation in t1_anisotropy_analysis.py
        R1_csa = (larmor_frequency_Hz**2) * J_omega_0 * 1e-12
        
        T1 = 1.0 / R1_csa
        
        self.R1_csa = R1_csa
        
        if self.config.verbose:
            print(f"\n  CSA T1 calculation (universal formula, any η):")
            print(f"    ω₀: {omega_0:.2e} rad/s ({larmor_frequency_Hz:.2e} Hz)")
            print(f"    J(ω₀): {J_omega_0:.2e} s")
            print(f"    R1_CSA: {R1_csa:.3f} s⁻¹")
            print(f"    Note: Correlation function must be in ppm² units")
        
        return T1
    
    def _calculate_T1_CSA_uniaxial(self, J_values: Dict[str, float]) -> float:
        """
        Calculate T1 for uniaxial CSA relaxation (η = 0) using analytical formula.
        
        R1 = (1/T1) = (1/3) × (f₀ × Δσ)² × J(ω₀)
        
        where f₀ is Larmor frequency in Hz and Δσ is in ppm (as dimensionless).
        
        This is the analytical formula from Lipari-Szabo theory for uniaxial CSA.
        Only valid for axially symmetric CSA tensor (η = 0).
        
        IMPORTANT: This formula expects J(ω) from NORMALIZED correlation functions
        (where C(0) = 1). The CSA magnitude (Δσ) is applied separately in the formula.
        Do NOT use this with correlation functions that already contain CSA in ppm -
        use _calculate_T1_CSA_universal() instead for that case.
        
        For correlation functions from trajectory simulations with arbitrary η,
        use _calculate_T1_CSA_universal() instead.
        
        Parameters
        ----------
        J_values : Dict[str, float]
            Spectral density at required frequencies (from normalized correlation)
        
        Returns
        -------
        T1 : float
            Longitudinal relaxation time (seconds)
        """
        omega_0 = self.config.get_omega0()  # rad/s
        larmor_freq_Hz = omega_0 / (2 * np.pi)  # Hz
        delta_sigma = self.config.delta_sigma if hasattr(self.config, 'delta_sigma') else 100.0  # ppm
        
        # Δσ in ppm is dimensionless (parts per million)
        # Convert to absolute by: Δσ_abs = Δσ_ppm × 10⁻⁶ × f₀_Hz
        delta_sigma_abs_Hz = delta_sigma * 1e-6 * larmor_freq_Hz  # Hz
        
        # R1_CSA = (1/3) × (f₀ × Δσ_ppm × 10⁻⁶)² × J(ω₀)
        # where f₀ is in Hz, Δσ_ppm is dimensionless
        # This gives proper units: (1/3) × Hz² × 10⁻¹² × s = s⁻¹
        J_omega_0 = J_values['J_omega']
        
        R1_csa = (1.0/3.0) * (delta_sigma_abs_Hz)**2 * J_omega_0
        
        T1 = 1.0 / R1_csa
        
        self.R1_csa = R1_csa
        
        if self.config.verbose:
            print(f"\n  CSA T1 calculation (analytical, η=0 only):")
            print(f"    f₀: {larmor_freq_Hz:.2e} Hz (ω₀ = {omega_0:.2e} rad/s)")
            print(f"    Δσ: {delta_sigma} ppm = {delta_sigma_abs_Hz:.2e} Hz")
            print(f"    J(ω₀): {J_omega_0:.2e}")
            print(f"    R1_CSA: {R1_csa:.3f} s⁻¹")
        
        return T1
    
    def _calculate_T1_dipolar(self, J_values: Dict[str, float]) -> float:
        """
        Calculate T1 for dipolar relaxation (e.g., 15N-1H).
        
        R1 = (d²/4) × [J(ωH-ωN) + 3J(ωN) + 6J(ωH+ωN)]
        
        where d = (μ₀/4π) × (γH × γN × ℏ) / r³NH
        
        Parameters
        ----------
        J_values : Dict[str, float]
            Spectral density at required frequencies
        
        Returns
        -------
        T1 : float
            Longitudinal relaxation time (seconds)
        """
        # Dipolar coupling constant
        d = self._calculate_dipolar_constant()
        
        # R1 = (d²/4) × [J(ωH-ωX) + 3J(ωX) + 6J(ωH+ωX)]
        J_diff = J_values['J_omega_H_minus_X']
        J_nucleus = J_values['J_omega']
        J_sum = J_values['J_omega_H_plus_X']
        
        R1_dipolar = (d**2 / 4.0) * (J_diff + 3*J_nucleus + 6*J_sum)
        
        T1 = 1.0 / R1_dipolar
        
        self.R1_dipolar = R1_dipolar
        
        if self.config.verbose:
            print(f"\n  Dipolar T1 calculation:")
            print(f"    d: {d:.2e} rad/s")
            print(f"    J(ωH-ωX): {J_diff:.2e}")
            print(f"    J(ωX): {J_nucleus:.2e}")
            print(f"    J(ωH+ωX): {J_sum:.2e}")
            print(f"    R1_dipolar: {R1_dipolar:.3f} s⁻¹")
        
        return T1
    
    def _calculate_T2_CSA(self, J_values: Dict[str, float]) -> float:
        """
        Calculate T2 for CSA relaxation.
        
        R2 = (1/15) × (ωN × Δσ)² × [4J(0) + 3J(ωN) + 6J(2ωN)]
        
        Parameters
        ----------
        J_values : Dict[str, float]
            Spectral density values
        
        Returns
        -------
        T2 : float
            Transverse relaxation time (seconds)
        """
        omega_0 = self.config.get_omega0()
        delta_sigma = self.config.delta_sigma if hasattr(self.config, 'delta_sigma') else 100.0
        delta_sigma_rad = delta_sigma * 1e-6 * omega_0
        
        J_0 = J_values['J_0']
        J_omega_0 = J_values['J_omega']
        
        R2_csa = (1.0/15.0) * (delta_sigma_rad)**2 * (4*J_0 + 3*J_omega_0)
        
        T2 = 1.0 / R2_csa
        
        self.R2_csa = R2_csa
        
        if self.config.verbose:
            print(f"\n  CSA T2 calculation:")
            print(f"    J(0): {J_0:.2e}")
            print(f"    J(ω₀): {J_omega_0:.2e}")
            print(f"    R2_CSA: {R2_csa:.3f} s⁻¹")
        
        return T2
    
    def _calculate_T2_dipolar(self, J_values: Dict[str, float]) -> float:
        """
        Calculate T2 for dipolar relaxation.
        
        R2 = (d²/8) × [4J(0) + J(ωH-ωN) + 3J(ωN) + 6J(ωH) + 6J(ωH+ωN)]
        
        Parameters
        ----------
        J_values : Dict[str, float]
            Spectral density values
        
        Returns
        -------
        T2 : float
            Transverse relaxation time (seconds)
        """
        d = self._calculate_dipolar_constant()
        
        J_0 = J_values['J_0']
        J_diff = J_values['J_omega_H_minus_X']
        J_nucleus = J_values['J_omega']
        J_H = J_values['J_omega_H']
        J_sum = J_values['J_omega_H_plus_X']
        
        R2_dipolar = (d**2 / 8.0) * (4*J_0 + J_diff + 3*J_nucleus + 6*J_H + 6*J_sum)
        
        T2 = 1.0 / R2_dipolar
        
        self.R2_dipolar = R2_dipolar
        
        if self.config.verbose:
            print(f"\n  Dipolar T2 calculation:")
            print(f"    J(0): {J_0:.2e}")
            print(f"    R2_dipolar: {R2_dipolar:.3f} s⁻¹")
        
        return T2
    
    def _calculate_dipolar_constant(self) -> float:
        """
        Calculate dipolar coupling constant d.
        
        d = (μ₀/4π) × (γH × γN × ℏ) / r³NH
        
        Returns
        -------
        d : float
            Dipolar coupling constant (rad/s)
        """
        # Physical constants
        mu_0 = 4 * np.pi * 1e-7  # T²m³/J
        hbar = 1.054571817e-34   # J·s
        
        # Gyromagnetic ratios (rad/T/s)
        gamma_H = GAMMA['1H']
        gamma_N = GAMMA[self.config.nucleus]
        
        # NH distance (typical for backbone)
        r_NH = 1.02e-10  # meters (1.02 Å)
        if hasattr(self.config, 'r_NH'):
            r_NH = self.config.r_NH
        
        # Dipolar constant
        d = -(mu_0 / (4 * np.pi)) * (gamma_H * gamma_N * hbar) / (r_NH**3)
        
        if self.config.verbose:
            print(f"\n  Dipolar coupling constant:")
            print(f"    r_NH: {r_NH*1e10:.2f} Å")
            print(f"    γH: {gamma_H:.2e} rad/(T·s)")
            print(f"    γN: {gamma_N:.2e} rad/(T·s)")
            print(f"    d: {d:.2e} rad/s ({d/(2*np.pi):.2e} Hz)")
        
        return d
    
    def calculate_NOE(self, J_values: Dict[str, float]) -> float:
        """
        Calculate heteronuclear NOE.
        
        NOE = 1 + (γH/γN) × (d²/4R1) × [6J(ωH+ωN) - J(ωH-ωN)]
        
        Parameters
        ----------
        J_values : Dict[str, float]
            Spectral density values
        
        Returns
        -------
        NOE : float
            Nuclear Overhauser Effect
        """
        if self.config.interaction_type != 'dipolar':
            if self.config.verbose:
                print("  ⚠️  NOE calculation only meaningful for dipolar relaxation")
            return 1.0
        
        gamma_H = GAMMA['1H']
        gamma_X = GAMMA[self.config.nucleus]
        
        d = self._calculate_dipolar_constant()
        
        J_diff = J_values['J_omega_H_minus_X']
        J_sum = J_values['J_omega_H_plus_X']
        
        # Need R1 for NOE
        if self.R1_dipolar is None:
            R1 = (d**2 / 4.0) * (J_diff + 3*J_values['J_omega'] + 6*J_sum)
        else:
            R1 = self.R1_dipolar
        
        NOE = 1 + (gamma_H / gamma_X) * (d**2 / (4 * R1)) * (6*J_sum - J_diff)
        
        self.NOE = NOE
        
        if self.config.verbose:
            print(f"\n  NOE calculation:")
            print(f"    NOE: {NOE:.3f}")
        
        return NOE
    
    def save(self, filepath: str):
        """
        Save NMR parameters.
        
        Parameters
        ----------
        filepath : str
            Output file path (.npz format)
        """
        save_dict = {
            'config': str(self.config.to_dict())
        }
        
        if self.T1 is not None:
            save_dict['T1'] = self.T1
            save_dict['R1_csa'] = self.R1_csa if self.R1_csa else 0
            save_dict['R1_dipolar'] = self.R1_dipolar if self.R1_dipolar else 0
        
        if self.T2 is not None:
            save_dict['T2'] = self.T2
            save_dict['R2_csa'] = self.R2_csa if self.R2_csa else 0
            save_dict['R2_dipolar'] = self.R2_dipolar if self.R2_dipolar else 0
        
        if self.NOE is not None:
            save_dict['NOE'] = self.NOE
        
        np.savez(filepath, **save_dict)
        
        if self.config.verbose:
            print(f"  ✓ Saved NMR parameters to: {filepath}")


# Example usage
if __name__ == '__main__':
    from .config import NMRConfig
    from .xyz_generator import TrajectoryGenerator
    from .euler_converter import EulerConverter
    from .spherical_harmonics import SphericalHarmonicsCalculator
    from .autocorrelation import AutocorrelationCalculator
    from .spectral_density import SpectralDensityCalculator
    
    # Test CSA
    config = NMRConfig(
        trajectory_type='diffusion_cone',
        S2=0.85,
        tau_c=5e-9,
        dt=1e-12,
        num_steps=10000,
        B0=14.1,
        nucleus='15N',
        interaction_type='CSA',
        delta_sigma=160.0,
        eta=0.0,
        calculate_T1=True,
        calculate_T2=True,
        verbose=True
    )
    
    # Full pipeline
    gen = TrajectoryGenerator(config)
    rotations, _ = gen.generate()
    
    converter = EulerConverter(config)
    euler_angles = converter.convert(rotations=rotations)
    
    sh_calc = SphericalHarmonicsCalculator(config)
    Y2m = sh_calc.calculate(euler_angles)
    
    acf_calc = AutocorrelationCalculator(config)
    acf, time_lags = acf_calc.calculate(Y2m)
    
    sd_calc = SpectralDensityCalculator(config)
    J, freq = sd_calc.calculate(acf, time_lags)
    
    # Calculate T1 and T2
    nmr_calc = NMRParametersCalculator(config)
    T1, T2 = nmr_calc.calculate(J, freq)
    
    print(f"\n{'='*50}")
    print(f"Results:")
    print(f"  T1 = {T1*1000:.1f} ms")
    print(f"  T2 = {T2*1000:.1f} ms")
