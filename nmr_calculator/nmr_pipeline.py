"""
Module 8: NMR Calculation Pipeline

Global orchestrator that combines all modules to calculate NMR parameters 
from molecular trajectories.

Pipeline steps:
1. Generate/load trajectory → xyz_generator
2. Convert to Euler angles → euler_converter  
3. Calculate Y₂ₘ → spherical_harmonics
4. Calculate ACF → autocorrelation
5. Calculate J(ω) → spectral_density
6. Calculate T1, T2 → nmr_parameters

Provides options to save intermediate results at each step.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple
from .config import NMRConfig
from .xyz_generator import TrajectoryGenerator
from .euler_converter import EulerConverter
from .spherical_harmonics import SphericalHarmonicsCalculator
from .autocorrelation import AutocorrelationCalculator
from .spectral_density import SpectralDensityCalculator
from .nmr_parameters import NMRParametersCalculator


class NMRPipeline:
    """
    Complete pipeline for NMR parameter calculations.
    
    Attributes
    ----------
    config : NMRConfig
        Configuration object controlling all calculations
    results : Dict
        Dictionary storing intermediate results from each step
    """
    
    def __init__(self, config: NMRConfig):
        """
        Initialize NMR calculation pipeline.
        
        Parameters
        ----------
        config : NMRConfig
            Configuration object
        """
        self.config = config
        self.results = {}
        
        # Create output directory
        if config.output_dir:
            Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        if config.verbose:
            print(f"\n{'='*70}")
            print("NMR CALCULATION PIPELINE")
            print(f"{'='*70}")
            self.config.summary()
            print(f"{'='*70}\n")
    
    def run(self, 
            coordinates: Optional[np.ndarray] = None,
            skip_steps: Optional[list] = None) -> Dict:
        """
        Run the complete NMR calculation pipeline.
        
        Parameters
        ----------
        coordinates : np.ndarray, optional
            Input coordinates (n_steps, n_atoms, 3)
            If None, generates trajectory according to config
        skip_steps : list, optional
            List of step names to skip (for debugging)
        
        Returns
        -------
        results : Dict
            Dictionary with all calculated quantities:
            - 'rotations': Rotation objects or None
            - 'euler_angles': Euler angles (n_steps, 3)
            - 'Y2m_coefficients': Y₂ₘ coefficients (n_steps, 5)
            - 'acf': Autocorrelation function
            - 'time_lags': ACF time lags
            - 'spectral_density': J(ω)
            - 'frequencies': Frequency axis
            - 'T1': Longitudinal relaxation time
            - 'T2': Transverse relaxation time (if calculated)
            - 'NOE': Nuclear Overhauser Effect (if applicable)
        """
        skip_steps = skip_steps or []
        
        # Step 1: Generate/Load Trajectory
        if 'trajectory' not in skip_steps:
            rotations, coords = self._step1_trajectory(coordinates)
            self.results['rotations'] = rotations
            self.results['coordinates'] = coords
        
        # Step 2: Convert to Euler Angles
        if 'euler' not in skip_steps:
            euler_angles = self._step2_euler_angles()
            self.results['euler_angles'] = euler_angles
        
        # Step 3: Calculate Y₂ₘ Coefficients
        if 'spherical_harmonics' not in skip_steps:
            Y2m = self._step3_spherical_harmonics()
            self.results['Y2m_coefficients'] = Y2m
        
        # Step 4: Calculate Autocorrelation Function
        if 'autocorrelation' not in skip_steps:
            acf, time_lags = self._step4_autocorrelation()
            self.results['acf'] = acf
            self.results['time_lags'] = time_lags
        
        # Step 5: Calculate Spectral Density
        if 'spectral_density' not in skip_steps:
            J, frequencies, freq_markers = self._step5_spectral_density()
            self.results['spectral_density'] = J
            self.results['frequencies'] = frequencies
            self.results['frequency_markers'] = freq_markers
        
        # Step 6: Calculate NMR Parameters (T1, T2)
        if 'nmr_parameters' not in skip_steps:
            T1, T2, NOE = self._step6_nmr_parameters()
            self.results['T1'] = T1
            self.results['T2'] = T2
            self.results['NOE'] = NOE
        
        if self.config.verbose:
            print(f"\n{'='*70}")
            print("PIPELINE COMPLETE")
            print(f"{'='*70}")
            self._print_summary()
        
        return self.results
    
    def _step1_trajectory(self, coordinates: Optional[np.ndarray]) -> Tuple:
        """
        Step 1: Generate or load molecular trajectory.
        
        Parameters
        ----------
        coordinates : np.ndarray, optional
            Input coordinates
        
        Returns
        -------
        rotations : list or None
            Rotation objects
        coordinates : np.ndarray or None
            Coordinates
        """
        gen = TrajectoryGenerator(self.config)
        
        if coordinates is not None:
            # Use provided coordinates
            if self.config.verbose:
                print("Using provided coordinates")
            rotations = None
            coords = coordinates
        else:
            # Generate new trajectory
            rotations, coords = gen.generate()
        
        # Save if requested
        if self.config.save_intermediate and self.config.output_dir:
            filepath = Path(self.config.output_dir) / "step1_trajectory.npz"
            gen.save(str(filepath))
        
        return rotations, coords
    
    def _step2_euler_angles(self) -> np.ndarray:
        """
        Step 2: Convert to Euler angles.
        
        Returns
        -------
        euler_angles : np.ndarray
            Euler angles (n_steps, 3)
        """
        converter = EulerConverter(self.config)
        
        # Use rotations if available, otherwise coordinates
        if 'rotations' in self.results and self.results['rotations'] is not None:
            euler_angles = converter.convert(rotations=self.results['rotations'])
        elif 'coordinates' in self.results and self.results['coordinates'] is not None:
            euler_angles = converter.convert(coordinates=self.results['coordinates'])
        else:
            raise ValueError("No trajectory data available")
        
        # Save if requested
        if self.config.save_intermediate and self.config.output_dir:
            filepath = Path(self.config.output_dir) / "step2_euler_angles.npz"
            converter.save(str(filepath))
        
        return euler_angles
    
    def _step3_spherical_harmonics(self) -> np.ndarray:
        """
        Step 3: Calculate Y₂ₘ spherical harmonics coefficients.
        
        Returns
        -------
        Y2m : np.ndarray
            Y₂ₘ coefficients (n_steps, 5)
        """
        sh_calc = SphericalHarmonicsCalculator(self.config)
        Y2m = sh_calc.calculate(self.results['euler_angles'])
        
        # Save if requested
        if self.config.save_intermediate and self.config.output_dir:
            filepath = Path(self.config.output_dir) / "step3_Y2m_coefficients.npz"
            sh_calc.save(str(filepath))
        
        return Y2m
    
    def _step4_autocorrelation(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Step 4: Calculate autocorrelation function.
        
        Returns
        -------
        acf : np.ndarray
            Autocorrelation function
        time_lags : np.ndarray
            Time lag values
        """
        acf_calc = AutocorrelationCalculator(self.config)
        acf, time_lags = acf_calc.calculate(self.results['Y2m_coefficients'])
        
        # Save if requested
        if self.config.save_intermediate and self.config.output_dir:
            filepath = Path(self.config.output_dir) / "step4_autocorrelation.npz"
            acf_calc.save(str(filepath))
        
        return acf, time_lags
    
    def _step5_spectral_density(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Step 5: Calculate spectral density function.
        
        Returns
        -------
        J : np.ndarray
            Spectral density
        frequencies : np.ndarray
            Frequency values
        freq_markers : Dict
            J(ω) at specific frequencies
        """
        sd_calc = SpectralDensityCalculator(self.config)
        J, frequencies = sd_calc.calculate(self.results['acf'], 
                                          self.results['time_lags'])
        
        # Save if requested
        if self.config.save_intermediate and self.config.output_dir:
            filepath = Path(self.config.output_dir) / "step5_spectral_density.npz"
            sd_calc.save(str(filepath))
        
        return J, frequencies, sd_calc.frequency_markers
    
    def _step6_nmr_parameters(self) -> Tuple[float, Optional[float], Optional[float]]:
        """
        Step 6: Calculate NMR relaxation parameters.
        
        Returns
        -------
        T1 : float
            Longitudinal relaxation time
        T2 : float or None
            Transverse relaxation time
        NOE : float or None
            Nuclear Overhauser Effect
        """
        nmr_calc = NMRParametersCalculator(self.config)
        
        T1, T2 = nmr_calc.calculate(
            self.results['spectral_density'],
            self.results['frequencies'],
            self.results.get('frequency_markers')
        )
        
        # Calculate NOE if applicable
        NOE = None
        if self.config.interaction_type == 'dipolar':
            # Get J values for NOE
            J_values = nmr_calc._get_J_at_frequencies(
                self.results['spectral_density'],
                self.results['frequencies']
            )
            NOE = nmr_calc.calculate_NOE(J_values)
        
        # Save if requested
        if self.config.save_intermediate and self.config.output_dir:
            filepath = Path(self.config.output_dir) / "step6_nmr_parameters.npz"
            nmr_calc.save(str(filepath))
        
        return T1, T2, NOE
    
    def _print_summary(self):
        """Print summary of calculated parameters."""
        print("\nFinal Results:")
        print(f"  {'='*50}")
        
        if 'T1' in self.results and self.results['T1']:
            print(f"  T1: {self.results['T1']:.3f} s ({self.results['T1']*1000:.1f} ms)")
        
        if 'T2' in self.results and self.results['T2']:
            print(f"  T2: {self.results['T2']:.3f} s ({self.results['T2']*1000:.1f} ms)")
        
        if 'NOE' in self.results and self.results['NOE']:
            print(f"  NOE: {self.results['NOE']:.3f}")
        
        print(f"  {'='*50}")
        
        if self.config.output_dir:
            print(f"\n  Results saved to: {self.config.output_dir}")
    
    def save_all_results(self, filepath: str):
        """
        Save all pipeline results to a single file.
        
        Parameters
        ----------
        filepath : str
            Output file path (.npz)
        """
        save_dict = {
            'config': str(self.config.to_dict())
        }
        
        # Add all numerical results
        for key, value in self.results.items():
            if isinstance(value, (np.ndarray, float, int)):
                save_dict[key] = value
            elif value is None:
                continue
            elif key == 'frequency_markers':
                # Special handling for dict
                if value:
                    marker_names = list(value.keys())
                    marker_omegas = [v[0] for v in value.values()]
                    marker_J_values = [v[1] for v in value.values()]
                    save_dict['marker_names'] = marker_names
                    save_dict['marker_omegas'] = marker_omegas
                    save_dict['marker_J_values'] = marker_J_values
        
        np.savez(filepath, **save_dict)
        
        if self.config.verbose:
            print(f"\n  ✓ Saved all results to: {filepath}")
    
    @staticmethod
    def load_results(filepath: str) -> Dict:
        """
        Load previously saved pipeline results.
        
        Parameters
        ----------
        filepath : str
            Path to saved results (.npz)
        
        Returns
        -------
        results : Dict
            Dictionary with all results
        """
        data = np.load(filepath, allow_pickle=True)
        results = {key: data[key] for key in data.files}
        
        return results


# Example usage
if __name__ == '__main__':
    from .config import NMRConfig
    
    # Configure calculation
    config = NMRConfig(
        # Trajectory
        trajectory_type='diffusion_cone',
        S2=0.85,
        tau_c=5e-9,
        dt=1e-12,
        num_steps=10000,
        
        # NMR parameters
        B0=14.1,
        nucleus='15N',
        interaction_type='CSA',
        delta_sigma=160.0,
        eta=0.0,
        
        # Calculation options
        max_lag=5000,
        lag_step=10,
        zero_fill_factor=4,
        frequency_markers=True,
        
        # Output
        calculate_T1=True,
        calculate_T2=True,
        output_dir='nmr_results',
        save_intermediate=True,
        verbose=True
    )
    
    # Run pipeline
    pipeline = NMRPipeline(config)
    results = pipeline.run()
    
    # Save all results
    pipeline.save_all_results('nmr_results/complete_results.npz')
    
    # Print results
    print(f"\n{'='*70}")
    print("Pipeline completed successfully!")
    print(f"T1 = {results['T1']*1000:.1f} ms")
    if results['T2']:
        print(f"T2 = {results['T2']*1000:.1f} ms")
