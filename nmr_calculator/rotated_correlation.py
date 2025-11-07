"""
Module 5: Rotated Correlation Function Matrix

Calculate rotated correlation matrices using Wigner D-matrices for ensemble averaging.
This module applies rotational transformations to correlation matrices for NMR relaxation
calculations accounting for molecular tumbling in solution.

Rotated Correlation Matrix:
    C'(τ) = D × C(τ) × D†
    
where D is the Wigner D-matrix and D† is its Hermitian conjugate.

For ensemble averaging:
    ⟨C'(τ)⟩ = (1/N) Σᵢ Dᵢ × C(τ) × Dᵢ†
"""

import numpy as np
from typing import Optional, Union, Tuple, Dict
import os
from .config import NMRConfig

# Try to import numba for optimization, fallback to numpy if unavailable
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Fallback: use numpy without numba optimization
    def njit(*args, **kwargs):
        """Dummy decorator when numba is not available"""
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    
    # prange falls back to range
    prange = range


class RotatedCorrelationCalculator:
    """
    Calculate rotated correlation matrices for ensemble-averaged NMR relaxation.
    
    This class performs the rotation of correlation matrices using Wigner D-matrices,
    which accounts for molecular tumbling in the lab frame. Can either load a
    pre-computed library of D-matrices or compute them from Euler angles.
    
    Attributes
    ----------
    config : NMRConfig
        Configuration object
    wigner_d_lib : np.ndarray or None
        Library of Wigner D-matrices, shape (n_orientations, 5, 5)
    rotated_correlations : np.ndarray or None
        Rotated correlation matrices, shape (n_orientations, 5, 5, n_lags)
    ensemble_avg : np.ndarray or None
        Ensemble-averaged correlation matrix, shape (5, 5, n_lags)
    """
    
    def __init__(self, config: NMRConfig):
        """
        Initialize rotated correlation calculator.
        
        Parameters
        ----------
        config : NMRConfig
            Configuration object with optional wigner_d_library path
        """
        self.config = config
        self.wigner_d_lib = None
        self.rotated_correlations = None
        self.ensemble_avg = None
        
    def load_wigner_d_library(self, library_path: str) -> bool:
        """
        Load pre-computed Wigner D-matrix library.
        
        The library should be an .npz file containing:
        - 'd2_matrix': Wigner D-matrices (n_orientations, 5, 5)
        
        Parameters
        ----------
        library_path : str
            Path to .npz file with Wigner D-matrices
            
        Returns
        -------
        success : bool
            True if loaded successfully
            
        Notes
        -----
        The reference uses: wigner_d_order2_N5000.npz with shape (5000, 5, 5)
        """
        try:
            if self.config.verbose:
                print(f"\n  Loading Wigner D-matrix library from:")
                print(f"    {library_path}")
            
            data = np.load(library_path, allow_pickle=True)
            self.wigner_d_lib = data['d2_matrix']
            
            if self.config.verbose:
                print(f"  ✓ Loaded Wigner D-matrix library")
                print(f"    Shape: {self.wigner_d_lib.shape}")
                print(f"    Number of orientations: {self.wigner_d_lib.shape[0]}")
                print(f"    Matrix size: {self.wigner_d_lib.shape[1]}×{self.wigner_d_lib.shape[2]}")
            
            return True
            
        except Exception as e:
            if self.config.verbose:
                print(f"  ⚠️  Failed to load library: {e}")
            return False
    
    def compute_wigner_d_from_euler(self, euler_angles: np.ndarray) -> np.ndarray:
        """
        Compute Wigner D-matrices from Euler angles.
        
        Alternative to loading a pre-computed library. Useful when you have
        specific orientations from MD trajectories.
        
        Parameters
        ----------
        euler_angles : np.ndarray
            Euler angles in ZYZ convention, shape (n_orientations, 3)
            Columns: [alpha, beta, gamma] in radians
            
        Returns
        -------
        wigner_d : np.ndarray
            Wigner D-matrices, shape (n_orientations, 5, 5)
            
        Notes
        -----
        This uses the WignerDCalculator class to compute D-matrices.
        For large ensembles, consider pre-computing and saving a library.
        """
        from .rotation_matrix import WignerDCalculator
        
        if self.config.verbose:
            print(f"\n  Computing Wigner D-matrices from Euler angles")
            print(f"    Number of orientations: {euler_angles.shape[0]}")
        
        wigner_calc = WignerDCalculator(self.config)
        wigner_d = wigner_calc.calculate_wigner_d_matrices(euler_angles)
        
        self.wigner_d_lib = wigner_d
        
        if self.config.verbose:
            print(f"  ✓ Computed Wigner D-matrices")
            print(f"    Shape: {wigner_d.shape}")
        
        return wigner_d
    
    def rotate_correlation_matrix(self,
                                  correlation_matrix: Dict[Tuple[int, int], np.ndarray],
                                  save_individual: bool = False,
                                  save_dir: Optional[str] = None) -> np.ndarray:
        """
        Rotate correlation matrix using Wigner D-matrices for all orientations.
        
        Applies the transformation:
            C'ₘ₁ₘ₂(τ) = Σₘ₃ₘ₄ Dₘ₁ₘ₃ × Cₘ₃ₘ₄(τ) × D*ₘ₂ₘ₄
        
        This is equivalent to:
            C'(τ) = D × C(τ) × D†
        
        Parameters
        ----------
        correlation_matrix : dict
            Correlation matrix from AutocorrelationCalculator.compute_correlation_matrix()
            Keys: (m1, m2) tuples where m1, m2 ∈ {-2, -1, 0, 1, 2}
            Values: Correlation arrays, shape (n_lags,)
        save_individual : bool, default=False
            If True, save each rotated matrix to disk
        save_dir : str, optional
            Directory to save individual matrices (required if save_individual=True)
            
        Returns
        -------
        rotated_corrs : np.ndarray
            Rotated correlation matrices, shape (n_orientations, 5, 5, n_lags)
            
        Notes
        -----
        The reference implementation uses the `rotate_all` function with numba
        parallel optimization for speed.
        
        If save_individual is True, saves files as:
            {save_dir}/rotated_corr_orientation_{i:05d}.npz
        """
        if self.wigner_d_lib is None:
            raise ValueError("No Wigner D-matrix library loaded. "
                           "Use load_wigner_d_library() or compute_wigner_d_from_euler() first.")
        
        if self.config.verbose:
            print(f"\n{'='*70}")
            print("MODULE 5: Rotating Correlation Matrices")
            print(f"{'='*70}")
            print(f"  Number of orientations: {self.wigner_d_lib.shape[0]}")
            print(f"  Correlation matrix size: 5×5")
        
        # Convert dict to array: (5, 5, n_lags)
        m_values = [-2, -1, 0, 1, 2]
        sample_corr = correlation_matrix[(0, 0)]
        n_lags = len(sample_corr)
        
        A = np.zeros((5, 5, n_lags), dtype=np.complex128)
        for i, m1 in enumerate(m_values):
            for j, m2 in enumerate(m_values):
                A[i, j, :] = correlation_matrix[(m1, m2)]
        
        if self.config.verbose:
            print(f"  Number of lag points: {n_lags}")
            if HAS_NUMBA:
                print(f"  Rotating using optimized numba implementation...")
            else:
                print(f"  Rotating using numpy implementation (numba not available)...")
        
        # Rotate using optimized function
        rotated_corrs = _rotate_all_optimized(self.wigner_d_lib, A)
        
        if self.config.verbose:
            print(f"  ✓ Rotated correlations computed")
            print(f"    Output shape: {rotated_corrs.shape}")
        
        self.rotated_correlations = rotated_corrs
        
        # Save individual matrices if requested
        if save_individual:
            if save_dir is None:
                raise ValueError("save_dir must be provided when save_individual=True")
            
            self._save_individual_matrices(rotated_corrs, save_dir)
        
        return rotated_corrs
    
    def compute_ensemble_average(self,
                                rotated_corrs: Optional[np.ndarray] = None,
                                save_path: Optional[str] = None) -> np.ndarray:
        """
        Compute ensemble-averaged correlation matrix.
        
        Averages rotated correlation matrices over all orientations:
            ⟨C(τ)⟩ = (1/N) Σᵢ C'ᵢ(τ)
        
        Parameters
        ----------
        rotated_corrs : np.ndarray, optional
            Rotated correlation matrices (n_orientations, 5, 5, n_lags)
            If None, uses stored rotated_correlations
        save_path : str, optional
            Path to save ensemble-averaged matrix (.npz format)
            
        Returns
        -------
        ensemble_avg : np.ndarray
            Ensemble-averaged correlation matrix, shape (5, 5, n_lags)
            
        Notes
        -----
        The ensemble average accounts for isotropic tumbling in solution.
        This is crucial for calculating relaxation rates that match experimental
        measurements on solution-state NMR.
        """
        if rotated_corrs is None:
            if self.rotated_correlations is None:
                raise ValueError("No rotated correlations available. "
                               "Run rotate_correlation_matrix() first.")
            rotated_corrs = self.rotated_correlations
        
        if self.config.verbose:
            print(f"\n  Computing ensemble average over {rotated_corrs.shape[0]} orientations...")
        
        # Average over orientation axis (axis 0)
        ensemble_avg = np.mean(rotated_corrs, axis=0)
        
        if self.config.verbose:
            print(f"  ✓ Ensemble-averaged correlation matrix computed")
            print(f"    Shape: {ensemble_avg.shape}")
        
        self.ensemble_avg = ensemble_avg
        
        # Save if requested
        if save_path is not None:
            self._save_ensemble_average(ensemble_avg, save_path)
        
        return ensemble_avg
    
    def _save_individual_matrices(self, rotated_corrs: np.ndarray, save_dir: str):
        """
        Save individual rotated correlation matrices.
        
        Parameters
        ----------
        rotated_corrs : np.ndarray
            Rotated correlations (n_orientations, 5, 5, n_lags)
        save_dir : str
            Output directory
        """
        os.makedirs(save_dir, exist_ok=True)
        
        n_orientations = rotated_corrs.shape[0]
        
        if self.config.verbose:
            print(f"\n  Saving {n_orientations} individual rotated matrices to:")
            print(f"    {save_dir}/")
        
        for i in range(n_orientations):
            filepath = os.path.join(save_dir, f"rotated_corr_orientation_{i:05d}.npz")
            
            # Save real and imaginary parts separately for better compression
            np.savez_compressed(
                filepath,
                rotated_corr_real=rotated_corrs[i].real,
                rotated_corr_imag=rotated_corrs[i].imag,
                orientation_index=i
            )
            
            if self.config.verbose and (i + 1) % 500 == 0:
                print(f"    Progress: {i+1}/{n_orientations} ({(i+1)/n_orientations*100:.1f}%)")
        
        if self.config.verbose:
            print(f"  ✓ Saved all individual matrices")
    
    def _save_ensemble_average(self, ensemble_avg: np.ndarray, save_path: str):
        """
        Save ensemble-averaged correlation matrix.
        
        Parameters
        ----------
        ensemble_avg : np.ndarray
            Ensemble average (5, 5, n_lags)
        save_path : str
            Output file path
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        np.savez_compressed(
            save_path,
            ensemble_avg_real=ensemble_avg.real,
            ensemble_avg_imag=ensemble_avg.imag,
            n_orientations=self.wigner_d_lib.shape[0] if self.wigner_d_lib is not None else 0,
            config=str(self.config.to_dict())
        )
        
        if self.config.verbose:
            print(f"  ✓ Saved ensemble-averaged matrix to:")
            print(f"    {save_path}")
    
    def load_ensemble_average(self, load_path: str) -> np.ndarray:
        """
        Load previously computed ensemble-averaged correlation matrix.
        
        Parameters
        ----------
        load_path : str
            Path to .npz file
            
        Returns
        -------
        ensemble_avg : np.ndarray
            Ensemble-averaged correlation matrix (5, 5, n_lags)
        """
        data = np.load(load_path)
        ensemble_avg = data['ensemble_avg_real'] + 1j * data['ensemble_avg_imag']
        
        self.ensemble_avg = ensemble_avg
        
        if self.config.verbose:
            print(f"  ✓ Loaded ensemble-averaged matrix from:")
            print(f"    {load_path}")
            print(f"    Shape: {ensemble_avg.shape}")
            if 'n_orientations' in data:
                print(f"    Number of orientations used: {data['n_orientations']}")
        
        return ensemble_avg


@njit(parallel=True)
def _rotate_all_optimized(D_all: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    Rotate correlation matrix using Wigner D-matrices (optimized with numba).
    
    Applies: C'(τ) = D × C(τ) × D†
    
    Parameters
    ----------
    D_all : np.ndarray
        Wigner D-matrices, shape (n_orientations, 5, 5)
    A : np.ndarray
        Correlation matrix, shape (5, 5, n_lags)
        
    Returns
    -------
    out : np.ndarray
        Rotated correlation matrices, shape (n_orientations, 5, 5, n_lags)
        
    Notes
    -----
    This is the optimized version from the reference (t1_anisotropy_analysis.py).
    Uses numba @njit with parallel=True for speed.
    Manual matrix multiplication to handle complex numbers correctly with numba.
    """
    n = D_all.shape[0]
    n_frames = A.shape[2]
    out = np.empty((n, 5, 5, n_frames), dtype=np.complex128)

    for i in prange(n):
        D = D_all[i]
        D_H = D.T.conj()  # Hermitian conjugate
        
        for t in range(n_frames):
            A_t = A[:, :, t]
            
            # Manual matmul: temp = D @ A_t
            temp = np.zeros((5, 5), dtype=np.complex128)
            for r in range(5):
                for c in range(5):
                    s = 0j
                    for k in range(5):
                        s += D[r, k] * A_t[k, c]
                    temp[r, c] = s
            
            # Manual matmul: out[i, :, :, t] = temp @ D_H
            for r in range(5):
                for c in range(5):
                    s = 0j
                    for k in range(5):
                        s += temp[r, k] * D_H[k, c]
                    out[i, r, c, t] = s
    
    return out


# Convenience function matching reference API
def rotate_all(D_all: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    Rotate correlation matrix using Wigner D-matrices.
    
    Convenience function matching the reference implementation API.
    
    Parameters
    ----------
    D_all : np.ndarray
        Wigner D-matrices (n_orientations, 5, 5)
    A : np.ndarray
        Correlation matrix (5, 5, n_lags)
        
    Returns
    -------
    rotated_corrs : np.ndarray
        Rotated correlation matrices (n_orientations, 5, 5, n_lags)
    """
    return _rotate_all_optimized(D_all, A)


# Example usage
if __name__ == '__main__':
    from .config import NMRConfig
    from .xyz_generator import TrajectoryGenerator
    from .euler_converter import EulerConverter
    from .spherical_harmonics import SphericalHarmonicsCalculator
    from .autocorrelation import AutocorrelationCalculator
    
    # Configuration
    config = NMRConfig(
        trajectory_type='diffusion_cone',
        S2=0.85,
        tau_c=5e-9,
        dt=0.02e-9,
        num_steps=1000,
        interaction_type='CSA',
        delta_sigma=100.0,
        max_lag=500,
        lag_step=1,
        verbose=True
    )
    
    print("\n" + "="*70)
    print("MODULE 5 EXAMPLE: Rotated Correlation Matrices")
    print("="*70)
    
    # Generate trajectory
    gen = TrajectoryGenerator(config)
    rotations, vectors = gen.generate()
    
    # Euler angles
    converter = EulerConverter(config)
    euler_angles = converter.convert(rotations=rotations)
    
    # Calculate Y2m
    sph_calc = SphericalHarmonicsCalculator(config)
    Y2m = sph_calc.calculate(vectors)
    
    # Correlation matrix
    acf_calc = AutocorrelationCalculator(config)
    corr_matrix = acf_calc.compute_correlation_matrix(Y2m)
    
    # Rotated correlations
    rot_calc = RotatedCorrelationCalculator(config)
    
    # Option 1: Compute from Euler angles
    rot_calc.compute_wigner_d_from_euler(euler_angles)
    rotated_corrs = rot_calc.rotate_correlation_matrix(corr_matrix)
    
    # Ensemble average
    ensemble_avg = rot_calc.compute_ensemble_average()
    
    print("\n✓ Module 5 example completed successfully")
