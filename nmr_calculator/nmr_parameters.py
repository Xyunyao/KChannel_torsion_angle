"""
Module 7: NMR Relaxation Parameters

Calculate T1 and T2 relaxation times from spectral density.

For CSA relaxation:
R1 = (1/T1) = (2/15) × (γB₀Δσ)² × [J(ω0)](1+η²/3)

R2 = (1/T2) = (1/30) × (γB₀Δσ)² × [4J(0) + 3J(ω0)](1+η²/3)

Here J(w) is scalar, independent of m.


Numerical:

R1 = (1/T1) =  (γB₀)² × [J(ω0)], here J(w) is calculated from correlation functions, and contains CSA information.

R2 = (1/T2) =  (γB₀)² × [4J(0) + 3J(ω0)]/2, here J(w) is calculated from correlation functions, and contains CSA information.


For dipolar relaxation (15N-1H):
Lipari-Szabo:
R1 = (d²/4) × [J(ωH-ωN) + 3J(ωN) + 6J(ωH+ωN)]
R2 = (d²/8) × [4J(0) + J(ωH-ωN) + 3J(ωN) + 6J(ωH) + 6J(ωH+ωN)]

where d = (μ₀/4π) × (γH γN ℏ) / r³NH

Numeric (Redfield):
[Placeholder - to be implemented]

Note: All frequencies ω are in rad/s throughout this module.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from .config import NMRConfig, GAMMA


class NMRParametersCalculator:
    """
    Calculate NMR relaxation parameters (T1, T2, NOE, etc.).
    
    Supports two calculation methods:
    1. Lipari-Szabo (LS): Analytical formulas
    2. Numeric: Universal formulas from Redfield theory
    
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
    R2_csa : float
        CSA contribution to R2
    R2_dipolar : float
        Dipolar contribution to R2
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
                 frequency_markers: Optional[Dict] = None,
                 method: str = 'numeric') -> Tuple[float, Optional[float]]:
        """
        Calculate T1 and T2 from spectral density.
        
        Parameters
        ----------
        spectral_density : np.ndarray
            Spectral density J(ω) (n_freq,)
        frequencies : np.ndarray
            Angular frequencies in rad/s (n_freq,)
        frequency_markers : Dict, optional
            Pre-calculated J(ω) at specific frequencies
        method : str, default='numeric'
            Calculation method: 'LS' (Lipari-Szabo) or 'numeric' (universal)
        
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
            print(f"  Calculation method: {method}")
        
        # Get spectral density at required frequencies
        J_values = self._get_J_at_frequencies(spectral_density, frequencies, frequency_markers)
        
        # Calculate T1
        if self.config.calculate_T1:
            if self.config.interaction_type == 'CSA':
                if method.upper() == 'LS':
                    T1 = self._calculate_T1_CSA_LS(J_values)
                else:
                    T1 = self._calculate_T1_CSA_numeric(J_values)
            elif self.config.interaction_type == 'dipolar':
                if method.upper() == 'LS':
                    T1 = self._calculate_T1_dipolar_LS(J_values)
                else:
                    T1 = self._calculate_T1_dipolar_numeric(J_values)
            else:
                raise ValueError(f"Unknown interaction type: {self.config.interaction_type}")
            
            self.T1 = T1
            
            if self.config.verbose:
                print(f"  ✓ T1 = {T1:.3f} s ({T1*1000:.1f} ms)")
        
        # Calculate T2
        T2 = None
        if self.config.calculate_T2:
            if self.config.interaction_type == 'CSA':
                if method.upper() == 'LS':
                    T2 = self._calculate_T2_CSA_LS(J_values)
                else:
                    T2 = self._calculate_T2_CSA_numeric(J_values)
            elif self.config.interaction_type == 'dipolar':
                if method.upper() == 'LS':
                    T2 = self._calculate_T2_dipolar_LS(J_values)
                else:
                    T2 = self._calculate_T2_dipolar_numeric(J_values)
            
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
        
        All frequencies are in rad/s.
        
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
        
        # Get Larmor frequencies (rad/s)
        omega_nucleus = self.config.get_omega0()  # rad/s - Observed nucleus (e.g., 15N)
        omega_H = 2 * np.pi * self.config.B0 * GAMMA['1H']  # rad/s - Proton
        
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
    
    # =========================================================================
    # CSA Relaxation - Lipari-Szabo (Analytical)
    # =========================================================================
    
    def _calculate_T1_CSA_LS(self, J_values: Dict[str, float]) -> float:
        """
        Calculate T1 for CSA relaxation using Lipari-Szabo formula.
        
        Lipari-Szabo (uniaxial CSA, η=0):
        R1 = (2/15) × (ω₀Δσ)² × J(ω₀)
        
        where:
        - ω₀ is Larmor frequency (rad/s)
        - Δσ is CSA in ppm (dimensionless, multiply by 10⁻⁶)
        - J(ω₀) is spectral density at Larmor frequency
        
        Valid only for axially symmetric CSA tensor (η = 0).
        For normalized correlation functions (C(0)=1).
        
        Parameters
        ----------
        J_values : Dict[str, float]
            Spectral density at required frequencies (from normalized ACF)
        
        Returns
        -------
        T1 : float
            Longitudinal relaxation time (seconds)
        """
        omega_0 = self.config.get_omega0()  # rad/s
        delta_sigma = self.config.delta_sigma if hasattr(self.config, 'delta_sigma') else 100.0  # ppm
        
        # Convert Δσ from ppm to absolute frequency: Δσ_rad = Δσ_ppm × 10⁻⁶ × ω₀
        delta_sigma_rad = delta_sigma * 1e-6 * omega_0  # rad/s
        
        J_omega_0 = J_values['J_omega']
        
        # R1 = (2/15) × (ω₀Δσ)² × J(ω₀)
        # Units: (rad/s)² × s = rad²/s = s⁻¹ ✓
        R1_csa = (2.0/15.0) * (delta_sigma_rad)**2 * J_omega_0
        
        T1 = 1.0 / R1_csa
        
        self.R1_csa = R1_csa
        
        if self.config.verbose:
            print(f"\n  CSA T1 calculation (Lipari-Szabo, η=0):")
            print(f"    ω₀: {omega_0:.2e} rad/s ({omega_0/(2*np.pi):.2e} Hz)")
            print(f"    Δσ: {delta_sigma} ppm = {delta_sigma_rad:.2e} rad/s")
            print(f"    J(ω₀): {J_omega_0:.2e} s")
            print(f"    R1_CSA: {R1_csa:.3f} s⁻¹")
            print(f"    Note: Valid only for η=0, normalized ACF")
        
        return T1
    
    def _calculate_T2_CSA_LS(self, J_values: Dict[str, float]) -> float:
        """
        Calculate T2 for CSA relaxation using Lipari-Szabo formula.
        
        Lipari-Szabo (uniaxial CSA, η=0):
        R2 = (1/30) × (ω₀Δσ)² × [4J(0) + 3J(ω₀)]
        
        where:
        - ω₀ is Larmor frequency (rad/s)
        - Δσ is CSA in ppm (dimensionless)
        - J(0), J(ω₀) are spectral densities
        
        Valid only for axially symmetric CSA tensor (η = 0).
        
        Parameters
        ----------
        J_values : Dict[str, float]
            Spectral density values
        
        Returns
        -------
        T2 : float
            Transverse relaxation time (seconds)
        """
        omega_0 = self.config.get_omega0()  # rad/s
        delta_sigma = self.config.delta_sigma if hasattr(self.config, 'delta_sigma') else 100.0  # ppm
        delta_sigma_rad = delta_sigma * 1e-6 * omega_0  # rad/s
        
        J_0 = J_values['J_0']
        J_omega_0 = J_values['J_omega']
        
        # R2 = (1/30) × (ω₀Δσ)² × [4J(0) + 3J(ω₀)]
        R2_csa = (1.0/30.0) * (delta_sigma_rad)**2 * (4*J_0 + 3*J_omega_0)
        
        T2 = 1.0 / R2_csa
        
        self.R2_csa = R2_csa
        
        if self.config.verbose:
            print(f"\n  CSA T2 calculation (Lipari-Szabo, η=0):")
            print(f"    J(0): {J_0:.2e} s")
            print(f"    J(ω₀): {J_omega_0:.2e} s")
            print(f"    R2_CSA: {R2_csa:.3f} s⁻¹")
        
        return T2
    
    # =========================================================================
    # CSA Relaxation - Numeric (Universal)
    # =========================================================================
    
    def _calculate_T1_CSA_numeric(self, J_values: Dict[str, float]) -> float:
        """
        Calculate T1 for CSA relaxation using numeric/universal formula.
        
        Numeric (from Redfield theory, any η):
        R1 = ω₀² × J(ω₀) × 10⁻¹²
        
        where:
        - ω₀ is Larmor frequency (rad/s)
        - J(ω₀) is spectral density from correlation function in ppm² units
        - 10⁻¹² accounts for ppm² conversion
        
        This formula works for ANY CSA tensor (any η value) when the correlation 
        function is calculated from Y2m spherical harmonics with CSA in ppm.
        
        IMPORTANT: This formula expects J(ω) from correlation functions that
        ALREADY CONTAIN the CSA magnitude in ppm. Do NOT use normalized ACF.
        
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
        
        J_omega_0 = J_values['J_omega']
        
        # R1 = ω₀² × J(ω₀) × 10⁻¹²
        # Units: (rad/s)² × s × 10⁻¹² = rad² × 10⁻¹² / s = s⁻¹ ✓
        R1_csa = (omega_0**2) * J_omega_0 * 1e-12
        
        T1 = 1.0 / R1_csa
        
        self.R1_csa = R1_csa
        
        if self.config.verbose:
            print(f"\n  CSA T1 calculation (Numeric/Universal, any η):")
            print(f"    ω₀: {omega_0:.2e} rad/s ({omega_0/(2*np.pi):.2e} Hz)")
            print(f"    J(ω₀): {J_omega_0:.2e} s")
            print(f"    R1_CSA: {R1_csa:.3f} s⁻¹")
            print(f"    Note: Correlation function must be in ppm² units")
        
        return T1
    
    def _calculate_T2_CSA_numeric(self, J_values: Dict[str, float]) -> float:
        """
        Calculate T2 for CSA relaxation using numeric/universal formula.
        
        Numeric (from Redfield theory, any η):
        [PLACEHOLDER - To be implemented]
        
        The formula should be derived from Redfield relaxation matrix elements
        for arbitrary CSA tensor symmetry.
        
        Temporarily using Lipari-Szabo formula as fallback.
        
        Parameters
        ----------
        J_values : Dict[str, float]
            Spectral density values
        
        Returns
        -------
        T2 : float
            Transverse relaxation time (seconds)
        """
        if self.config.verbose:
            print(f"\n  ⚠️  CSA T2 numeric method not yet implemented")
            print(f"    Using Lipari-Szabo formula as fallback...")
        
        # Fallback to LS formula
        return self._calculate_T2_CSA_LS(J_values)
    
    # =========================================================================
    # Dipolar Relaxation - Lipari-Szabo (Analytical)
    # =========================================================================
    
    def _calculate_T1_dipolar_LS(self, J_values: Dict[str, float]) -> float:
        """
        Calculate T1 for dipolar relaxation using Lipari-Szabo formula.
        
        Lipari-Szabo (heteronuclear dipolar, e.g., 15N-1H):
        R1 = (d²/4) × [J(ωH-ωN) + 3J(ωN) + 6J(ωH+ωN)]
        
        where d = (μ₀/4π) × (γH γN ℏ) / r³NH
        
        All frequencies ω are in rad/s.
        
        Parameters
        ----------
        J_values : Dict[str, float]
            Spectral density at required frequencies
        
        Returns
        -------
        T1 : float
            Longitudinal relaxation time (seconds)
        """
        # Dipolar coupling constant (rad/s)
        d = self._calculate_dipolar_constant()
        
        # Extract spectral densities
        J_diff = J_values['J_omega_H_minus_X']  # J(ωH - ωN)
        J_nucleus = J_values['J_omega']  # J(ωN)
        J_sum = J_values['J_omega_H_plus_X']  # J(ωH + ωN)
        
        # R1 = (d²/4) × [J(ωH-ωN) + 3J(ωN) + 6J(ωH+ωN)]
        # Units: (rad/s)² × s = rad²/s = s⁻¹ ✓
        R1_dipolar = (d**2 / 4.0) * (J_diff + 3*J_nucleus + 6*J_sum)
        
        T1 = 1.0 / R1_dipolar
        
        self.R1_dipolar = R1_dipolar
        
        if self.config.verbose:
            print(f"\n  Dipolar T1 calculation (Lipari-Szabo):")
            print(f"    d: {d:.2e} rad/s ({d/(2*np.pi):.2e} Hz)")
            print(f"    J(ωH-ωN): {J_diff:.2e} s")
            print(f"    J(ωN): {J_nucleus:.2e} s")
            print(f"    J(ωH+ωN): {J_sum:.2e} s")
            print(f"    R1_dipolar: {R1_dipolar:.3f} s⁻¹")
        
        return T1
    
    def _calculate_T2_dipolar_LS(self, J_values: Dict[str, float]) -> float:
        """
        Calculate T2 for dipolar relaxation using Lipari-Szabo formula.
        
        Lipari-Szabo (heteronuclear dipolar, e.g., 15N-1H):
        R2 = (d²/8) × [4J(0) + J(ωH-ωN) + 3J(ωN) + 6J(ωH) + 6J(ωH+ωN)]
        
        where d = (μ₀/4π) × (γH γN ℏ) / r³NH
        
        All frequencies ω are in rad/s.
        
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
        
        J_0 = J_values['J_0']  # J(0)
        J_diff = J_values['J_omega_H_minus_X']  # J(ωH - ωN)
        J_nucleus = J_values['J_omega']  # J(ωN)
        J_H = J_values['J_omega_H']  # J(ωH)
        J_sum = J_values['J_omega_H_plus_X']  # J(ωH + ωN)
        
        # R2 = (d²/8) × [4J(0) + J(ωH-ωN) + 3J(ωN) + 6J(ωH) + 6J(ωH+ωN)]
        R2_dipolar = (d**2 / 8.0) * (4*J_0 + J_diff + 3*J_nucleus + 6*J_H + 6*J_sum)
        
        T2 = 1.0 / R2_dipolar
        
        self.R2_dipolar = R2_dipolar
        
        if self.config.verbose:
            print(f"\n  Dipolar T2 calculation (Lipari-Szabo):")
            print(f"    J(0): {J_0:.2e} s")
            print(f"    J(ωH-ωN): {J_diff:.2e} s")
            print(f"    J(ωN): {J_nucleus:.2e} s")
            print(f"    J(ωH): {J_H:.2e} s")
            print(f"    J(ωH+ωN): {J_sum:.2e} s")
            print(f"    R2_dipolar: {R2_dipolar:.3f} s⁻¹")
        
        return T2
    
    # =========================================================================
    # Dipolar Relaxation - Numeric (Universal)
    # =========================================================================
    
    def _calculate_T1_dipolar_numeric(self, J_values: Dict[str, float]) -> float:
        """
        Calculate T1 for dipolar relaxation using numeric/universal formula.
        
        Numeric (from Redfield theory):
        [PLACEHOLDER - To be implemented]
        
        Should be derived from Redfield relaxation matrix for dipolar interaction
        using rotated correlation matrix C(m,m',τ).
        
        Temporarily using Lipari-Szabo formula as fallback.
        
        Parameters
        ----------
        J_values : Dict[str, float]
            Spectral density at required frequencies
        
        Returns
        -------
        T1 : float
            Longitudinal relaxation time (seconds)
        """
        if self.config.verbose:
            print(f"\n  ⚠️  Dipolar T1 numeric method not yet implemented")
            print(f"    Using Lipari-Szabo formula as fallback...")
        
        # Fallback to LS formula
        return self._calculate_T1_dipolar_LS(J_values)
    
    def _calculate_T2_dipolar_numeric(self, J_values: Dict[str, float]) -> float:
        """
        Calculate T2 for dipolar relaxation using numeric/universal formula.
        
        Numeric (from Redfield theory):
        [PLACEHOLDER - To be implemented]
        
        Should be derived from Redfield relaxation matrix for dipolar interaction
        using rotated correlation matrix C(m,m',τ).
        
        Temporarily using Lipari-Szabo formula as fallback.
        
        Parameters
        ----------
        J_values : Dict[str, float]
            Spectral density values
        
        Returns
        -------
        T2 : float
            Transverse relaxation time (seconds)
        """
        if self.config.verbose:
            print(f"\n  ⚠️  Dipolar T2 numeric method not yet implemented")
            print(f"    Using Lipari-Szabo formula as fallback...")
        
        # Fallback to LS formula
        return self._calculate_T2_dipolar_LS(J_values)
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _calculate_dipolar_constant(self) -> float:
        """
        Calculate dipolar coupling constant d.
        
        d = (μ₀/4π) × (γH × γN × ℏ) / r³NH
        
        All quantities in SI units.
        
        Returns
        -------
        d : float
            Dipolar coupling constant (rad/s)
        """
        # Physical constants (SI units)
        mu_0 = 4 * np.pi * 1e-7  # T²m³/J (permeability of free space)
        hbar = 1.054571817e-34   # J·s (reduced Planck constant)
        
        # Gyromagnetic ratios (rad/(T·s))
        gamma_H = GAMMA['1H']
        gamma_N = GAMMA[self.config.nucleus]
        
        # NH distance (typical for backbone)
        r_NH = 1.02e-10  # meters (1.02 Å)
        if hasattr(self.config, 'r_NH'):
            r_NH = self.config.r_NH
        
        # Dipolar constant
        # d = -(μ₀/4π) × (γH × γN × ℏ) / r³NH
        # Units: (T²m³/J) × (rad/(T·s))² × (J·s) / m³ = rad/s ✓
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
    
    # Test CSA with both methods
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
    
    # Calculate T1 and T2 using Lipari-Szabo
    print("\n" + "="*70)
    print("LIPARI-SZABO METHOD")
    print("="*70)
    nmr_calc_LS = NMRParametersCalculator(config)
    T1_LS, T2_LS = nmr_calc_LS.calculate(J, freq, method='LS')
    
    # Calculate T1 and T2 using Numeric
    print("\n" + "="*70)
    print("NUMERIC/UNIVERSAL METHOD")
    print("="*70)
    nmr_calc_numeric = NMRParametersCalculator(config)
    T1_numeric, T2_numeric = nmr_calc_numeric.calculate(J, freq, method='numeric')
    
    print(f"\n{'='*70}")
    print(f"Comparison:")
    print(f"  Lipari-Szabo:  T1 = {T1_LS*1000:.1f} ms, T2 = {T2_LS*1000:.1f} ms")
    print(f"  Numeric:       T1 = {T1_numeric*1000:.1f} ms, T2 = {T2_numeric*1000:.1f} ms")
    print(f"{'='*70}")
