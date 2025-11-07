"""
Module 5: Rotation Matrix and Wigner-D Functions

Calculate Wigner-D rotation matrices for spherical harmonics rotations.

For rank-2 spherical harmonics:
D²ₘₘ'(α,β,γ) = exp(-imα) × d²ₘₘ'(β) × exp(-im'γ)

Where d²ₘₘ'(β) are reduced Wigner rotation matrix elements.

Used to transform spherical harmonics between reference frames.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Optional, Union, Tuple
from .config import NMRConfig


class WignerDCalculator:
    """
    Calculate Wigner-D rotation matrices and apply to spherical harmonics.
    
    Attributes
    ----------
    config : NMRConfig
        Configuration object
    wigner_d_matrices : np.ndarray
        Wigner-D matrices for each time step (n_steps, 5, 5)
        Indices: [time_step, m, m']
    """
    
    def __init__(self, config: NMRConfig):
        """
        Initialize Wigner-D calculator.
        
        Parameters
        ----------
        config : NMRConfig
            Configuration object
        """
        self.config = config
        self.wigner_d_matrices = None
        self.ensemble_averaged_acf = None
        
    def calculate_wigner_d_matrices(self, euler_angles: np.ndarray) -> np.ndarray:
        """
        Calculate Wigner-D matrices for rank-2 from Euler angles.
        
        Parameters
        ----------
        euler_angles : np.ndarray
            Euler angles in ZYZ convention (n_steps, 3)
        
        Returns
        -------
        wigner_d : np.ndarray
            Wigner-D matrices (n_steps, 5, 5)
            Indices correspond to m, m' = -2, -1, 0, 1, 2
        """
        if self.config.verbose:
            print(f"\n{'='*70}")
            print("MODULE 5: Calculating Wigner-D Rotation Matrices")
            print(f"{'='*70}")
            print(f"  Rank: 2 (for NMR interactions)")
            print(f"  Matrix size: 5×5 (m = -2, -1, 0, 1, 2)")
        
        n_steps = euler_angles.shape[0]
        wigner_d = np.zeros((n_steps, 5, 5), dtype=complex)
        
        # Extract Euler angles
        alpha = euler_angles[:, 0]
        beta = euler_angles[:, 1]
        gamma = euler_angles[:, 2]
        
        # Calculate for each time step
        for i in range(n_steps):
            if self.config.verbose and i % 1000 == 0 and i > 0:
                print(f"  Progress: {i}/{n_steps} ({i/n_steps*100:.1f}%)", end='\r')
            
            wigner_d[i] = self._wigner_d_rank2(alpha[i], beta[i], gamma[i])
        
        if self.config.verbose:
            print(f"\n  ✓ Calculated {n_steps} Wigner-D matrices")
        
        self.wigner_d_matrices = wigner_d
        return wigner_d
    
    def _wigner_d_rank2(self, alpha: float, beta: float, gamma: float) -> np.ndarray:
        """
        Calculate rank-2 Wigner-D matrix for given Euler angles.
        
        D²ₘₘ'(α,β,γ) = exp(-imα) × d²ₘₘ'(β) × exp(-im'γ)
        
        Parameters
        ----------
        alpha, beta, gamma : float
            Euler angles (ZYZ convention)
        
        Returns
        -------
        D : np.ndarray
            5×5 Wigner-D matrix (complex)
        """
        # Calculate reduced rotation matrix d²(β)
        d = self._wigner_d_small_rank2(beta)
        
        # Apply phase factors
        m_values = np.array([-2, -1, 0, 1, 2])
        phase_alpha = np.exp(-1j * m_values * alpha)
        phase_gamma = np.exp(-1j * m_values * gamma)
        
        # D²ₘₘ'(α,β,γ) = exp(-imα) × d²ₘₘ'(β) × exp(-im'γ)
        D = phase_alpha[:, np.newaxis] * d * phase_gamma[np.newaxis, :]
        
        return D
    
    def _wigner_d_small_rank2(self, beta: float) -> np.ndarray:
        """
        Calculate reduced Wigner rotation matrix d²(β) for rank-2.
        
        These are the angular-dependent parts of Wigner-D matrices.
        Uses formulas from Varshalovich et al., "Quantum Theory of Angular Momentum".
        
        Parameters
        ----------
        beta : float
            Euler angle β (polar angle)
        
        Returns
        -------
        d : np.ndarray
            5×5 reduced rotation matrix (real)
            
        Notes
        -----
        Fixed formulas for d²₋₁,₋₁(β) and d²₁,₁(β) to give correct identity at β=0.
        The formulas use: d²ₘ,ₘ'(β) with proper symmetries and normalizations.
        """
        cb = np.cos(beta)
        sb = np.sin(beta)
        c2 = np.cos(beta / 2)
        s2 = np.sin(beta / 2)
        
        d = np.zeros((5, 5))
        
        # m' = -2
        d[0, 0] = c2**4  # m = -2
        d[1, 0] = 2 * c2**3 * s2  # m = -1
        d[2, 0] = np.sqrt(6) * c2**2 * s2**2  # m = 0
        d[3, 0] = 2 * c2 * s2**3  # m = 1
        d[4, 0] = s2**4  # m = 2
        
        # m' = -1
        d[0, 1] = -2 * c2**3 * s2  # m = -2
        d[1, 1] = c2**2 * (2 * cb - 1)  # m = -1, FIXED: was (2*cb + 1)
        d[2, 1] = np.sqrt(6) * c2 * s2 * cb  # m = 0
        d[3, 1] = s2**2 * (2 * cb + 1)  # m = 1, FIXED: removed negative sign
        d[4, 1] = 2 * s2**3 * c2  # m = 2
        
        # m' = 0
        d[0, 2] = np.sqrt(6) * c2**2 * s2**2  # m = -2
        d[1, 2] = -np.sqrt(6) * c2 * s2 * cb  # m = -1
        d[2, 2] = (3 * cb**2 - 1) / 2  # m = 0
        d[3, 2] = np.sqrt(6) * c2 * s2 * cb  # m = 1
        d[4, 2] = np.sqrt(6) * c2**2 * s2**2  # m = 2
        
        # m' = 1
        d[0, 3] = -2 * c2 * s2**3  # m = -2
        d[1, 3] = s2**2 * (2 * cb + 1)  # m = -1, FIXED: was (2*cb - 1)
        d[2, 3] = -np.sqrt(6) * c2 * s2 * cb  # m = 0
        d[3, 3] = c2**2 * (2 * cb - 1)  # m = 1, FIXED: was (2*cb + 1)
        d[4, 3] = 2 * c2**3 * s2  # m = 2
        
        # m' = 2
        d[0, 4] = s2**4  # m = -2
        d[1, 4] = -2 * c2 * s2**3  # m = -1
        d[2, 4] = np.sqrt(6) * c2**2 * s2**2  # m = 0
        d[3, 4] = -2 * c2**3 * s2  # m = 1
        d[4, 4] = c2**4  # m = 2
        
        return d
    
    def apply_rotation_to_Y2m(self, 
                              Y2m_initial: np.ndarray,
                              wigner_d: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply Wigner-D rotation to Y₂ₘ coefficients.
        
        Y'₂ₘ = Σₘ' D²ₘₘ' × Y₂ₘ'
        
        Parameters
        ----------
        Y2m_initial : np.ndarray
            Initial Y₂ₘ coefficients (5,) or (n_steps, 5)
        wigner_d : np.ndarray, optional
            Wigner-D matrices (n_steps, 5, 5)
            If None, uses stored matrices
        
        Returns
        -------
        Y2m_rotated : np.ndarray
            Rotated Y₂ₘ coefficients
        """
        if wigner_d is None:
            if self.wigner_d_matrices is None:
                raise ValueError("No Wigner-D matrices available")
            wigner_d = self.wigner_d_matrices
        
        # Handle single vector or time series
        if Y2m_initial.ndim == 1:
            Y2m_initial = Y2m_initial[np.newaxis, :]
            single_vector = True
        else:
            single_vector = False
        
        n_steps = Y2m_initial.shape[0]
        Y2m_rotated = np.zeros_like(Y2m_initial)
        
        # Apply rotation: Y'ₘ = Σₘ' Dₘₘ' × Yₘ'
        for i in range(n_steps):
            Y2m_rotated[i] = wigner_d[i] @ Y2m_initial[i]
        
        if single_vector:
            Y2m_rotated = Y2m_rotated[0]
        
        return Y2m_rotated
    
    def calculate_ensemble_averaged_acf(self,
                                       Y2m_coefficients: np.ndarray,
                                       individual_rotations: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate ensemble-averaged ACF using Wigner-D rotation averaging.
        
        This accounts for rotational averaging in the lab frame.
        
        ⟨Y₂ₘ(0) × Y₂ₘ'(τ)⟩ = Σₘₘ' ⟨D²ₘₘ₁ × D²*ₘ'ₘ₂⟩ × Y₂ₘ₁ × Y₂ₘ₂
        
        Parameters
        ----------
        Y2m_coefficients : np.ndarray
            Y₂ₘ time series (n_steps, 5)
        individual_rotations : np.ndarray, optional
            Individual rotation trajectories for ensemble (n_ensemble, n_steps, 3, 3)
        
        Returns
        -------
        acf_ensemble : np.ndarray
            Ensemble-averaged ACF
        """
        if self.config.verbose:
            print(f"\n  Calculating ensemble-averaged ACF using Wigner-D")
        
        # Placeholder for ensemble averaging
        # This would require multiple trajectories
        
        if self.config.verbose:
            print("    ⚠️  Ensemble averaging not yet implemented")
            print("    Returning single-trajectory ACF")
        
        # For now, return simple ACF
        from .autocorrelation import AutocorrelationCalculator
        acf_calc = AutocorrelationCalculator(self.config)
        acf, _ = acf_calc.calculate(Y2m_coefficients)
        
        return acf
    
    @staticmethod
    def rotation_to_wigner_d(rotation: R, rank: int = 2) -> np.ndarray:
        """
        Convert scipy Rotation to Wigner-D matrix.
        
        Parameters
        ----------
        rotation : Rotation
            Scipy rotation object
        rank : int
            Angular momentum rank (default: 2 for NMR)
        
        Returns
        -------
        D : np.ndarray
            Wigner-D matrix ((2*rank+1), (2*rank+1))
        """
        # Get Euler angles
        euler = rotation.as_euler('ZYZ')
        
        # Calculate Wigner-D
        calc = WignerDCalculator(NMRConfig())
        D = calc._wigner_d_rank2(euler[0], euler[1], euler[2])
        
        return D
    
    def load_precomputed_library(self, library_path: str) -> bool:
        """
        Load pre-computed Wigner-D library for efficiency.
        
        Parameters
        ----------
        library_path : str
            Path to .npz file with pre-computed matrices
        
        Returns
        -------
        success : bool
            True if loaded successfully
        """
        try:
            data = np.load(library_path)
            self.wigner_d_matrices = data['wigner_d_matrices']
            
            if self.config.verbose:
                print(f"  ✓ Loaded Wigner-D library: {self.wigner_d_matrices.shape}")
            
            return True
        except Exception as e:
            if self.config.verbose:
                print(f"  ⚠️  Could not load library: {e}")
            return False
    
    def save(self, filepath: str):
        """
        Save Wigner-D matrices.
        
        Parameters
        ----------
        filepath : str
            Output file path (.npz format)
        """
        if self.wigner_d_matrices is None:
            raise ValueError("No Wigner-D matrices to save. Run calculate_wigner_d_matrices() first.")
        
        np.savez(filepath,
                wigner_d_matrices_real=self.wigner_d_matrices.real,
                wigner_d_matrices_imag=self.wigner_d_matrices.imag,
                config=str(self.config.to_dict()))
        
        if self.config.verbose:
            print(f"  ✓ Saved Wigner-D matrices to: {filepath}")


# Example usage
if __name__ == '__main__':
    from .config import NMRConfig
    from .xyz_generator import TrajectoryGenerator
    from .euler_converter import EulerConverter
    
    # Test
    config = NMRConfig(
        trajectory_type='diffusion_cone',
        S2=0.85,
        tau_c=2e-9,
        num_steps=100,
        verbose=True
    )
    
    # Generate trajectory
    gen = TrajectoryGenerator(config)
    rotations, _ = gen.generate()
    
    # Euler angles
    converter = EulerConverter(config)
    euler_angles = converter.convert(rotations=rotations)
    
    # Wigner-D
    wigner_calc = WignerDCalculator(config)
    wigner_d = wigner_calc.calculate_wigner_d_matrices(euler_angles)
    
    print(f"\nWigner-D shape: {wigner_d.shape}")
    print(f"First matrix (t=0):")
    print(wigner_d[0].real)
