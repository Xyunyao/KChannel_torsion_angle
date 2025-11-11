"""
Module 6: Spectral Density Function

Calculate spectral density J(ω) from autocorrelation function using FFT.

J(ω) = ∫₋∞^∞ C(τ) × exp(-iωτ) dτ
     = 2 × ∫₀^∞ C(τ) × cos(ωτ) dτ  (for real C(τ))

Includes zero-filling for improved frequency resolution and 
frequency markers for specific ω values (0, ωH, ωN, etc.).
"""

import numpy as np
from typing import Dict, Optional, Tuple, Union
from .config import NMRConfig


def moving_average(data: np.ndarray, window_size: int, axis: int = 0) -> np.ndarray:
    """
    Compute a moving average over a sliding window along a given axis.
    
    Uses convolution with a uniform kernel for smoothing.

    Parameters
    ----------
    data : np.ndarray
        Input data (1D or multi-dimensional).
    window_size : int
        Size of the moving window.
    axis : int, optional
        Axis along which to apply the moving average (default is 0).

    Returns
    -------
    np.ndarray
        Smoothed array of the same shape as input.
    """
    if window_size < 1:
        raise ValueError("window_size must be >= 1")

    data = np.asarray(data, dtype=float)
    kernel = np.ones(window_size) / window_size

    # Move the specified axis to the front, apply convolution, then restore original order
    data_swapped = np.moveaxis(data, axis, 0)
    smoothed = np.apply_along_axis(
        lambda m: np.convolve(m, kernel, mode='same'),
        axis=0,
        arr=data_swapped
    )
    return np.moveaxis(smoothed, 0, axis)


class SpectralDensityCalculator:
    """
    Calculate spectral density from autocorrelation function.
    
    Attributes
    ----------
    config : NMRConfig
        Configuration object
    spectral_density : np.ndarray
        Spectral density values J(ω) (n_freq,)
    frequencies : np.ndarray
        Frequency values (rad/s) (n_freq,)
    frequency_markers : Dict[str, Tuple[float, float]]
        J(ω) values at specific frequencies
        Keys: frequency names, Values: (ω, J(ω))
    """
    
    def __init__(self, config: NMRConfig):
        """
        Initialize spectral density calculator.
        
        Parameters
        ----------
        config : NMRConfig
            Configuration object
        """
        self.config = config
        self.spectral_density = None
        self.frequencies = None
        self.frequency_markers = {}
        
    def calculate(self, 
                 acf: np.ndarray, 
                 time_lags: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate spectral density from autocorrelation function.
        
        Parameters
        ----------
        acf : np.ndarray
            Autocorrelation function (n_lags,)
        time_lags : np.ndarray
            Time lag values in seconds (n_lags,)
        
        Returns
        -------
        spectral_density : np.ndarray
            Spectral density J(ω) (n_freq,)
        frequencies : np.ndarray
            Angular frequencies (rad/s) (n_freq,)
        """
        if self.config.verbose:
            print(f"\n{'='*70}")
            print("MODULE 6: Calculating Spectral Density")
            print(f"{'='*70}")
            print(f"  Method: FFT with cosine transform")
        
        # Get zero-fill factor
        zero_fill_factor = self.config.zero_fill_factor if hasattr(self.config, 'zero_fill_factor') and self.config.zero_fill_factor else 1
        
        if self.config.verbose:
            print(f"  ACF length: {len(acf)}")
            print(f"  Time range: 0 to {time_lags[-1]:.2e} s")
            print(f"  Zero-fill factor: {zero_fill_factor}×")
        
        # Subtract DC offset (average of last 100 points)
        n_offset_points = min(100, len(acf))
        dc_offset = np.mean(acf[-n_offset_points:])
        acf_corrected = acf - dc_offset
        
        if self.config.verbose:
            print(f"  DC offset (avg of last {n_offset_points} points): {dc_offset:.2e}")
        
        # Apply zero-filling
        if zero_fill_factor > 1:
            n_fill = int(len(acf_corrected) * (zero_fill_factor - 1))
            acf_filled = np.concatenate([acf_corrected, np.zeros(n_fill)])
            
            # Extend time axis
            dt = time_lags[1] - time_lags[0] if len(time_lags) > 1 else self.config.dt
            time_filled = np.arange(len(acf_filled)) * dt
        else:
            acf_filled = acf_corrected
            time_filled = time_lags
        
        if self.config.verbose and zero_fill_factor > 1:
            print(f"  After zero-fill: {len(acf_filled)} points")
        
        # Calculate spectral density using FFT
        spectral_density, frequencies = self._fft_spectral_density(acf_filled, time_filled)
        
        # Apply moving average smoothing
        smoothing_window = self.config.smoothing_window if hasattr(self.config, 'smoothing_window') else 5
        if smoothing_window > 1:
            spectral_density = moving_average(spectral_density, smoothing_window, axis=0)
            if self.config.verbose:
                print(f"  Applied moving average smoothing (window={smoothing_window})")
        
        # Calculate at specific frequency markers
        if self.config.frequency_markers:
            self._calculate_frequency_markers(spectral_density, frequencies)
        
        if self.config.verbose:
            print(f"  ✓ Calculated spectral density")
            print(f"  Frequency points: {len(frequencies)}")
            print(f"  Frequency range: {frequencies[0]:.2e} to {frequencies[-1]:.2e} rad/s")
            print(f"  J(0): {spectral_density[0]:.2e}")
            
            if self.frequency_markers:
                print(f"\n  Frequency markers:")
                for name, (omega, J_omega) in self.frequency_markers.items():
                    print(f"    {name}: ω={omega:.2e} rad/s, J(ω)={J_omega:.2e}")
        
        self.spectral_density = spectral_density
        self.frequencies = frequencies
        
        return spectral_density, frequencies
    
    def _fft_spectral_density(self, 
                             acf: np.ndarray, 
                             time_lags: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate spectral density using FFT.
        
        J(ω) = 2 × ∫₀^∞ C(τ) × cos(ωτ) dτ
             = 2 × Re[FFT{C(τ)}]
        
        Parameters
        ----------
        acf : np.ndarray
            Autocorrelation function (n_lags,)
        time_lags : np.ndarray
            Time lags (seconds) (n_lags,)
        
        Returns
        -------
        spectral_density : np.ndarray
            J(ω) (n_freq,)
        frequencies : np.ndarray
            Angular frequencies (rad/s) (n_freq,)
        """
        # Time step
        dt = time_lags[1] - time_lags[0] if len(time_lags) > 1 else self.config.dt
        
        # FFT of ACF
        fft_acf = np.fft.rfft(acf)
        
        # Spectral density (factor of 2 for positive frequencies only)
        spectral_density = 2 * dt * fft_acf.real
        
        # Frequency axis (angular frequency)
        n_freq = len(spectral_density)
        freq_hz = np.fft.rfftfreq(len(acf), dt)
        frequencies = 2 * np.pi * freq_hz  # Convert to rad/s
        
        return spectral_density, frequencies
    
    def _calculate_frequency_markers(self, 
                                    spectral_density: np.ndarray,
                                    frequencies: np.ndarray):
        """
        Calculate J(ω) at specific frequency markers.
        
        Markers typically include:
        - ω = 0
        - ω = ωH (proton Larmor frequency)
        - ω = ωN or ωC (heteronucleus Larmor frequency)
        - ω = ωH ± ωN (sum/difference frequencies)
        
        Parameters
        ----------
        spectral_density : np.ndarray
            Spectral density values
        frequencies : np.ndarray
            Frequency values (rad/s)
        """
        self.frequency_markers = {}
        
        # Get marker frequencies from config
        marker_frequencies = self.config.get_marker_frequencies()
        
        for name, omega in marker_frequencies.items():
            # Find nearest frequency point
            idx = np.argmin(np.abs(frequencies - omega))
            J_omega = spectral_density[idx]
            
            self.frequency_markers[name] = (omega, J_omega)
    
    def calculate_analytical_J(self, 
                              frequencies: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate analytical spectral density using Lipari-Szabo model.
        
        J(ω) = (2/5) × [S²τc/(1+(ωτc)²) + (1-S²)τe/(1+(ωτe)²)]
        
        where τe⁻¹ = τc⁻¹ + τf⁻¹ (τf is fast motion time scale)
        
        For simple model with single correlation time:
        J(ω) = (2/5) × τc/(1+(ωτc)²)
        
        Parameters
        ----------
        frequencies : np.ndarray, optional
            Frequency values (rad/s)
            If None, uses stored frequencies
        
        Returns
        -------
        J_analytical : np.ndarray
            Analytical spectral density
        """
        if frequencies is None:
            if self.frequencies is None:
                raise ValueError("No frequencies available")
            frequencies = self.frequencies
        
        tau_c = self.config.tau_c
        S2 = self.config.S2 if hasattr(self.config, 'S2') else 1.0
        
        # Simple Lipari-Szabo model
        J_analytical = (2.0 / 5.0) * S2 * tau_c / (1 + (frequencies * tau_c)**2)
        
        # Add fast motion contribution if (1 - S²) significant
        if S2 < 0.99:
            tau_f = tau_c / 100  # Assume fast motion 100× faster
            tau_e = 1 / (1/tau_c + 1/tau_f)
            J_analytical += (2.0 / 5.0) * (1 - S2) * tau_e / (1 + (frequencies * tau_e)**2)
        
        if self.config.verbose:
            print(f"\n  Analytical J(ω) (Lipari-Szabo):")
            print(f"    J(0): {J_analytical[0]:.2e}")
            print(f"    τ_c: {tau_c:.2e} s")
            print(f"    S²: {S2:.3f}")
        
        return J_analytical
    
    def compare_with_analytical(self) -> Tuple[float, float]:
        """
        Compare FFT spectral density with analytical Lipari-Szabo.
        
        Returns
        -------
        rmse : float
            Root mean square error
        max_error : float
            Maximum relative error
        """
        if self.spectral_density is None or self.frequencies is None:
            raise ValueError("No spectral density calculated")
        
        # Calculate analytical
        J_analytical = self.calculate_analytical_J(self.frequencies)
        
        # Calculate errors
        errors = np.abs(self.spectral_density - J_analytical)
        relative_errors = errors / np.maximum(J_analytical, 1e-20)
        
        rmse = np.sqrt(np.mean(errors**2))
        max_error = np.max(relative_errors)
        
        if self.config.verbose:
            print(f"\n  Comparison with analytical:")
            print(f"    RMSE: {rmse:.2e}")
            print(f"    Max relative error: {max_error:.2%}")
        
        return rmse, max_error
    
    @staticmethod
    def J_lipari_szabo(omega: Union[float, np.ndarray],
                      S2: float,
                      tau_c: float,
                      tau_f: Optional[float] = None) -> Union[float, np.ndarray]:
        """
        Lipari-Szabo spectral density function.
        
        Parameters
        ----------
        omega : float or np.ndarray
            Angular frequency (rad/s)
        S2 : float
            Order parameter (0 to 1)
        tau_c : float
            Overall correlation time (s)
        tau_f : float, optional
            Fast internal motion time (s)
            If None, uses single-exponential model
        
        Returns
        -------
        J : float or np.ndarray
            Spectral density
        """
        if tau_f is None:
            # Single exponential
            J = (2.0 / 5.0) * tau_c / (1 + (omega * tau_c)**2)
        else:
            # Two-component model
            tau_e = 1 / (1/tau_c + 1/tau_f)
            J = (2.0 / 5.0) * (S2 * tau_c / (1 + (omega * tau_c)**2) + 
                               (1 - S2) * tau_e / (1 + (omega * tau_e)**2))
        
        return J
    
    def save(self, filepath: str):
        """
        Save spectral density results.
        
        Parameters
        ----------
        filepath : str
            Output file path (.npz format)
        """
        if self.spectral_density is None:
            raise ValueError("No spectral density to save. Run calculate() first.")
        
        save_dict = {
            'spectral_density': self.spectral_density,
            'frequencies': self.frequencies,
            'config': str(self.config.to_dict())
        }
        
        # Add frequency markers
        if self.frequency_markers:
            marker_names = list(self.frequency_markers.keys())
            marker_omegas = [v[0] for v in self.frequency_markers.values()]
            marker_J_values = [v[1] for v in self.frequency_markers.values()]
            
            save_dict['marker_names'] = marker_names
            save_dict['marker_omegas'] = marker_omegas
            save_dict['marker_J_values'] = marker_J_values
        
        np.savez(filepath, **save_dict)
        
        if self.config.verbose:
            print(f"  ✓ Saved spectral density to: {filepath}")


# Example usage
if __name__ == '__main__':
    from .config import NMRConfig
    from .xyz_generator import TrajectoryGenerator
    from .euler_converter import EulerConverter
    from .spherical_harmonics import SphericalHarmonicsCalculator
    from .autocorrelation import AutocorrelationCalculator
    
    # Test
    config = NMRConfig(
        trajectory_type='diffusion_cone',
        S2=0.85,
        tau_c=2e-9,
        dt=1e-12,
        num_steps=10000,
        B0=14.1,
        nucleus='15N',
        interaction_type='CSA',
        delta_sigma=100.0,
        eta=0.0,
        zero_fill_factor=4,
        frequency_markers=True,
        verbose=True
    )
    
    # Pipeline
    gen = TrajectoryGenerator(config)
    rotations, _ = gen.generate()
    
    converter = EulerConverter(config)
    euler_angles = converter.convert(rotations=rotations)
    
    sh_calc = SphericalHarmonicsCalculator(config)
    Y2m = sh_calc.calculate(euler_angles)
    
    acf_calc = AutocorrelationCalculator(config)
    acf, time_lags = acf_calc.calculate(Y2m)
    
    # Spectral density
    sd_calc = SpectralDensityCalculator(config)
    J, freq = sd_calc.calculate(acf, time_lags)
    
    print(f"\nSpectral density shape: {J.shape}")
    print(f"Frequency range: {freq[0]:.2e} to {freq[-1]:.2e} rad/s")
    
    # Compare
    rmse, max_err = sd_calc.compare_with_analytical()
