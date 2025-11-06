"""
Module 3: Spherical Harmonics Decomposition

Decompose NMR interactions (CSA tensor or dipolar coupling) into rank-2 
spherical harmonics Y₂ₘ series using rigorous Wigner D-matrix formalism.

For CSA (Chemical Shift Anisotropy):
- Principal axis system (PAS) defined by δ_iso, Δδ, η or δ_xx, δ_yy, δ_zz
- Transforms to Y₂ₘ components in molecular frame via Wigner D-matrix rotation

For Dipolar Coupling:
- Axially symmetric (η = 0)
- Y₂₀ component along internuclear vector

Mathematical Framework:
- Uses full Wigner D-matrix for l=2 rotation
- Symbolic tensor definitions with sympy
- Supports arbitrary asymmetry parameter η
"""

import numpy as np
import sympy as sp
from sympy.physics.quantum.spin import Rotation
from scipy.spatial.transform import Rotation as R
from typing import Tuple, Optional, Dict

# Handle both relative and absolute imports
try:
    from .config import NMRConfig
except ImportError:
    from config import NMRConfig


class SphericalHarmonicsCalculator:
    """
    Calculate spherical harmonics coefficients for NMR interactions.
    
    Two implementations available:
    1. Sympy-based: Rigorous symbolic Wigner D-matrix (slower, mathematically explicit)
    2. NumPy-based: Direct numerical Wigner d-functions (faster, optimized)
    
    Attributes
    ----------
    config : NMRConfig
        Configuration object
    Y2m_coefficients : np.ndarray
        Y₂ₘ coefficients for each time step (n_steps, 5)
        Columns: [Y₂₋₂, Y₂₋₁, Y₂₀, Y₂₁, Y₂₂]
    use_sympy : bool
        If True, use sympy symbolic calculation (default: False for speed)
    """
    
    def __init__(self, config: NMRConfig, use_sympy: bool = False):
        """
        Initialize spherical harmonics calculator.
        
        Parameters
        ----------
        config : NMRConfig
            Configuration object
        use_sympy : bool, optional
            If True, use sympy symbolic calculation (slower but explicit)
            If False, use NumPy optimized calculation (faster, default)
        """
        self.config = config
        self.Y2m_coefficients = None
        self.use_sympy = use_sympy
        
        # Setup symbolic tensor definitions and Wigner D-matrix (if using sympy)
        if self.use_sympy:
            self._setup_symbolic_tensors()
            if self.config.verbose:
                print("Using sympy-based symbolic Wigner D-matrix calculation")
        else:
            if self.config.verbose:
                print("Using NumPy-optimized Wigner d-matrix calculation")
        
    def _setup_symbolic_tensors(self):
        """
        Setup symbolic tensor definitions for CSA using Wigner D-matrix formalism.
        
        This method creates symbolic expressions for:
        1. CSA tensor components in PAS: T_2m = {-2, -1, 0, 1, 2}
        2. Wigner D-matrix for l=2 rotation
        3. Transformed tensor in lab frame: D_2 * T_2m
        
        The symbolic expressions are later lambdified for numerical evaluation.
        
        Mathematical Details:
        --------------------
        CSA tensor in PAS (Principal Axis System):
            T_2^{-2} = (δ_xx - δ_yy) / 2
            T_2^{-1} = 0
            T_2^{0}  = sqrt(3/2) * (δ_zz - (δ_xx + δ_yy + δ_zz)/3)
            T_2^{1}  = 0
            T_2^{2}  = (δ_xx - δ_yy) / 2
        
        Wigner D-matrix elements (l=2):
            D_{m1,m2}^{(2)}(α,β,γ) = exp(-i*m1*α) * d_{m1,m2}^{(2)}(β) * exp(-i*m2*γ)
        
        where d_{m1,m2}^{(2)}(β) is the Wigner small-d matrix from sympy.physics.quantum.spin
        
        Transformation:
            Y_2^m(lab) = Σ_{m'} D_{m,m'}^{(2)} * T_2^{m'}(PAS)
        """
        # Define symbolic variables
        self.alpha, self.beta, self.gamma = sp.symbols('alpha beta gamma', real=True)
        self.delta_xx, self.delta_yy, self.delta_zz = sp.symbols('delta_xx delta_yy delta_zz', real=True)
        self.iso = sp.symbols('iso', real=True)
        
        # CSA tensor in PAS (5 components for l=2)
        self.CSA_T_2m = {
            '-2': sp.Rational(1, 2) * (self.delta_xx - self.delta_yy),
            '-1': sp.S(0),
            '0': sp.sqrt(sp.Rational(3, 2)) * (self.delta_zz - (self.delta_xx + self.delta_yy + self.delta_zz) / 3),
            '1': sp.S(0),
            '2': sp.Rational(1, 2) * (self.delta_xx - self.delta_yy)
        }
        
        # Construct full Wigner D-matrix for l=2
        m_values = [-2, -1, 0, 1, 2]
        self.D_2 = sp.zeros(5, 5)
        for i, m1 in enumerate(m_values):
            for j, m2 in enumerate(m_values):
                # D_{m1,m2}^{(2)} = exp(-i*m1*α) * d_{m1,m2}^{(2)}(β) * exp(-i*m2*γ)
                self.D_2[i, j] = (sp.exp(-sp.I * m1 * self.alpha) * 
                                  Rotation.d(2, m1, m2, self.beta) * 
                                  sp.exp(-sp.I * m2 * self.gamma))
        
        # Transform CSA tensor: Y_2^m = D_2 * T_2m
        T_2m_matrix = sp.Matrix([self.CSA_T_2m[str(m)] for m in m_values])
        self.CSA_transformed = self.D_2 * T_2m_matrix
        self.CSA_transformed = sp.simplify(self.CSA_transformed)
        
        # Lambdify for numerical evaluation (creates fast numpy functions)
        self.CSA_Y_lm_funcs = []
        for m in range(-2, 3):
            expr = self.CSA_transformed[m + 2]  # Index: m+2 maps [-2,2] to [0,4]
            func = sp.lambdify((self.alpha, self.beta, self.gamma, 
                               self.delta_xx, self.delta_yy, self.delta_zz), 
                              expr, modules='numpy')
            self.CSA_Y_lm_funcs.append(func)
        
        if self.config.verbose:
            print("Symbolic tensor setup complete:")
            print(f"  - CSA tensor T_2m defined in PAS")
            print(f"  - Wigner D-matrix (5×5) constructed for l=2")
            print(f"  - Transformation D_2 * T_2m computed symbolically")
            print(f"  - 5 lambda functions created for Y_2^m (m=-2...2)")

    @staticmethod
    def _wigner_d_matrix_l2(beta):
        """
        Calculate Wigner small-d matrix elements for l=2 (NumPy optimized).
        
        This is a fast numerical implementation of d^(2)_{m1,m2}(β) for all
        m1, m2 ∈ {-2, -1, 0, 1, 2}.
        
        Parameters
        ----------
        beta : np.ndarray
            Polar angle β (can be scalar or array)
            
        Returns
        -------
        d_matrix : np.ndarray
            Wigner d-matrix (5, 5) if beta is scalar, or (n_steps, 5, 5) if array
            
        Notes
        -----
        Formulas from Varshalovich et al., "Quantum Theory of Angular Momentum" (1988)
        and Edmonds, "Angular Momentum in Quantum Mechanics" (1996).
        
        Matrix indices: [m1+2, m2+2] where m1, m2 ∈ {-2, -1, 0, 1, 2}
        """
        # Ensure beta is array for vectorization
        beta = np.atleast_1d(beta)
        n_steps = len(beta)
        
        # Precompute trigonometric functions
        cos_b = np.cos(beta)
        sin_b = np.sin(beta)
        sin_b_2 = sin_b**2
        cos_2b = np.cos(2 * beta)
        sin_2b = np.sin(2 * beta)
        
        # Initialize d-matrix
        d = np.zeros((n_steps, 5, 5))
        
        # Fill d-matrix elements using CORRECT formulas from sympy
        # Formulas verified against sympy.physics.quantum.spin.Rotation.d(2, m1, m2, beta)
        
        # Row 0: m1 = -2
        d[:, 0, 0] = -sin_b_2/4 + cos_b/2 + 1/2  # d^2_{-2,-2}
        d[:, 0, 1] = (cos_b + 1) * sin_b / 2  # d^2_{-2,-1}
        d[:, 0, 2] = np.sqrt(6) * sin_b_2 / 4  # d^2_{-2,0}
        d[:, 0, 3] = (1 - cos_b) * sin_b / 2  # d^2_{-2,1}
        d[:, 0, 4] = -sin_b_2/4 - cos_b/2 + 1/2  # d^2_{-2,2}
        
        # Row 1: m1 = -1
        d[:, 1, 0] = -(cos_b + 1) * sin_b / 2  # d^2_{-1,-2}
        d[:, 1, 1] = cos_b/2 + cos_2b/2  # d^2_{-1,-1}
        d[:, 1, 2] = np.sqrt(6) * sin_2b / 4  # d^2_{-1,0}
        d[:, 1, 3] = cos_b/2 - cos_2b/2  # d^2_{-1,1}
        d[:, 1, 4] = (1 - cos_b) * sin_b / 2  # d^2_{-1,2}
        
        # Row 2: m1 = 0
        d[:, 2, 0] = np.sqrt(6) * sin_b_2 / 4  # d^2_{0,-2}
        d[:, 2, 1] = -np.sqrt(6) * sin_2b / 4  # d^2_{0,-1}
        d[:, 2, 2] = 1 - 3*sin_b_2/2  # d^2_{0,0}
        d[:, 2, 3] = np.sqrt(6) * sin_2b / 4  # d^2_{0,1}
        d[:, 2, 4] = np.sqrt(6) * sin_b_2 / 4  # d^2_{0,2}
        
        # Row 3: m1 = 1
        d[:, 3, 0] = (cos_b - 1) * sin_b / 2  # d^2_{1,-2}
        d[:, 3, 1] = cos_b/2 - cos_2b/2  # d^2_{1,-1}
        d[:, 3, 2] = -np.sqrt(6) * sin_2b / 4  # d^2_{1,0}
        d[:, 3, 3] = cos_b/2 + cos_2b/2  # d^2_{1,1}
        d[:, 3, 4] = (cos_b + 1) * sin_b / 2  # d^2_{1,2}
        
        # Row 4: m1 = 2
        d[:, 4, 0] = -sin_b_2/4 - cos_b/2 + 1/2  # d^2_{2,-2}
        d[:, 4, 1] = (cos_b - 1) * sin_b / 2  # d^2_{2,-1}
        d[:, 4, 2] = np.sqrt(6) * sin_b_2 / 4  # d^2_{2,0}
        d[:, 4, 3] = -(cos_b + 1) * sin_b / 2  # d^2_{2,1}
        d[:, 4, 4] = -sin_b_2/4 + cos_b/2 + 1/2  # d^2_{2,2}
        
        # Return single matrix if input was scalar
        if n_steps == 1:
            return d[0]
        return d
    
    @staticmethod
    def _calculate_wigner_D_matrix_l2_numpy(alpha, beta, gamma):
        """
        Calculate full Wigner D-matrix for l=2 using NumPy (fast version).
        
        D^(2)_{m1,m2}(α,β,γ) = exp(-i*m1*α) * d^(2)_{m1,m2}(β) * exp(-i*m2*γ)
        
        Parameters
        ----------
        alpha, beta, gamma : np.ndarray
            Euler angles (ZYZ convention), shape (n_steps,)
            
        Returns
        -------
        D_matrix : np.ndarray
            Full Wigner D-matrix (n_steps, 5, 5)
        """
        n_steps = len(alpha)
        m_values = np.array([-2, -1, 0, 1, 2])
        
        # Get d-matrix
        d_matrix = SphericalHarmonicsCalculator._wigner_d_matrix_l2(beta)
        if d_matrix.ndim == 2:
            d_matrix = d_matrix[np.newaxis, :, :]  # Add batch dimension
        
        # Initialize D-matrix
        D_matrix = np.zeros((n_steps, 5, 5), dtype=complex)
        
        # Apply phase factors: D = exp(-i*m1*α) * d * exp(-i*m2*γ)
        for i, m1 in enumerate(m_values):
            for j, m2 in enumerate(m_values):
                phase_alpha = np.exp(-1j * m1 * alpha)
                phase_gamma = np.exp(-1j * m2 * gamma)
                D_matrix[:, i, j] = phase_alpha * d_matrix[:, i, j] * phase_gamma
        
        return D_matrix
    
    def _transform_csa_tensor_numpy(self, alpha, beta, gamma, delta_xx, delta_yy, delta_zz):
        """
        Transform CSA tensor using NumPy-based Wigner D-matrix (fast version).
        
        Y_2^m = Σ_{m'} D^(2)_{m,m'}(α,β,γ) * T_2^{m'}
        
        Parameters
        ----------
        alpha, beta, gamma : np.ndarray
            Euler angles, shape (n_steps,)
        delta_xx, delta_yy, delta_zz : float
            CSA tensor components in PAS
            
        Returns
        -------
        Y2m : np.ndarray
            Transformed tensor, shape (n_steps, 5)
        """
        # CSA tensor in PAS
        T_2m = np.array([
            0.5 * (delta_xx - delta_yy),        # T_2^{-2}
            0.0,                                 # T_2^{-1}
            np.sqrt(3/2) * (delta_zz - (delta_xx + delta_yy + delta_zz) / 3),  # T_2^{0}
            0.0,                                 # T_2^{1}
            0.5 * (delta_xx - delta_yy)         # T_2^{2}
        ])
        
        # Get Wigner D-matrix
        D_matrix = self._calculate_wigner_D_matrix_l2_numpy(alpha, beta, gamma)
        
        # Transform: Y_2^m = D * T_2m
        # D_matrix: (n_steps, 5, 5), T_2m: (5,) -> result: (n_steps, 5)
        Y2m = np.einsum('nij,j->ni', D_matrix, T_2m)
        
        return Y2m


        
    def calculate(self, euler_angles: np.ndarray) -> np.ndarray:
        """
        Calculate Y₂ₘ coefficients from Euler angles.
        
        Parameters
        ----------
        euler_angles : np.ndarray
            Euler angles in ZYZ convention (n_steps, 3)
        
        Returns
        -------
        Y2m_coefficients : np.ndarray
            Y₂ₘ coefficients (n_steps, 5)
            Columns: [Y₂₋₂, Y₂₋₁, Y₂₀, Y₂₁, Y₂₂]
        """
        if self.config.verbose:
            print(f"\n{'='*70}")
            print("MODULE 3: Calculating Spherical Harmonics")
            print(f"{'='*70}")
            print(f"  Interaction type: {self.config.interaction_type}")
        
        if self.config.interaction_type == 'CSA':
            Y2m_coefficients = self._calculate_CSA(euler_angles)
        elif self.config.interaction_type == 'dipolar':
            Y2m_coefficients = self._calculate_dipolar(euler_angles)
        else:
            raise ValueError(f"Unknown interaction type: {self.config.interaction_type}")
        
        if self.config.verbose:
            print(f"  ✓ Calculated Y₂ₘ for {len(Y2m_coefficients)} time steps")
            print(f"  Shape: {Y2m_coefficients.shape}")
        
        self.Y2m_coefficients = Y2m_coefficients
        return Y2m_coefficients
    
    def _calculate_CSA(self, euler_angles: np.ndarray) -> np.ndarray:
        """
        Calculate Y₂ₘ coefficients for CSA interaction using full Wigner D-matrix formalism.
        
        This method uses rigorous quantum mechanical rotation formalism:
        1. Define CSA tensor in PAS: T_2m
        2. Construct Wigner D-matrix for l=2
        3. Transform: Y_2^m(lab) = Σ_{m'} D_{m,m'}^{(2)} * T_2^{m'}(PAS)
        
        CSA tensor in PAS:
            σ_PAS = [σ_11, σ_22, σ_33] or [σ_xx, σ_yy, σ_zz]
        
        Two parameterization options:
        
        Option 1 - Traditional NMR (Δδ, η):
            δ_iso = (σ_11 + σ_22 + σ_33) / 3
            Δδ = σ_33 - δ_iso  (anisotropy)
            η = (σ_22 - σ_11) / Δδ  (asymmetry, 0 ≤ η ≤ 1)
            
            Then:
            δ_zz = δ_iso + Δδ
            δ_xx = δ_iso - Δδ*(1+η)/2
            δ_yy = δ_iso - Δδ*(1-η)/2
        
        Option 2 - Direct tensor components (δ_xx, δ_yy, δ_zz):
            Use values directly from config
        
        Parameters
        ----------
        euler_angles : np.ndarray
            Euler angles (n_steps, 3) in ZYZ convention
            [:,0] = α (phi)
            [:,1] = β (theta)
            [:,2] = γ (psi)
        
        Returns
        -------
        Y2m_coefficients : np.ndarray
            Y₂ₘ coefficients (n_steps, 5)
            Columns: [Y₂₋₂, Y₂₋₁, Y₂₀, Y₂₁, Y₂₂]
        
        Notes
        -----
        This implementation supports arbitrary asymmetry parameter η, unlike
        simplified approximations that only work for axial (η=0) or special cases.
        """
        n_steps = euler_angles.shape[0]
        Y2m_coefficients = np.zeros((n_steps, 5), dtype=complex)
        
        # Extract Euler angles (ZYZ convention)
        alpha = euler_angles[:, 0]  # φ
        beta = euler_angles[:, 1]   # θ
        gamma = euler_angles[:, 2]  # ψ
        
        # Determine CSA tensor components
        # Check if direct tensor components are provided
        if (hasattr(self.config, 'delta_xx') and self.config.delta_xx is not None and
            hasattr(self.config, 'delta_yy') and self.config.delta_yy is not None and
            hasattr(self.config, 'delta_zz') and self.config.delta_zz is not None):
            # Option 2: Direct tensor components
            delta_xx_val = self.config.delta_xx
            delta_yy_val = self.config.delta_yy
            delta_zz_val = self.config.delta_zz
            
            if self.config.verbose:
                print(f"  CSA tensor (direct components):")
                print(f"    δ_xx: {delta_xx_val:.2f} ppm")
                print(f"    δ_yy: {delta_yy_val:.2f} ppm")
                print(f"    δ_zz: {delta_zz_val:.2f} ppm")
        else:
            # Option 1: Convert from Δδ and η
            delta_sigma = self.config.delta_sigma if hasattr(self.config, 'delta_sigma') else 100.0  # ppm
            eta = self.config.eta if hasattr(self.config, 'eta') else 0.0
            delta_iso = self.config.delta_iso if hasattr(self.config, 'delta_iso') else 0.0  # ppm
            
            # Convert to principal components
            # δ_zz is along unique axis
            delta_zz_val = delta_iso + delta_sigma
            # δ_xx and δ_yy in perpendicular plane
            delta_xx_val = delta_iso - delta_sigma * (1 + eta) / 2
            delta_yy_val = delta_iso - delta_sigma * (1 - eta) / 2
            
            if self.config.verbose:
                print(f"  CSA Parameters:")
                print(f"    δ_iso: {delta_iso:.2f} ppm")
                print(f"    Δδ (anisotropy): {delta_sigma:.2f} ppm")
                print(f"    η (asymmetry): {eta:.3f}")
                print(f"  Converted to tensor components:")
                print(f"    δ_xx: {delta_xx_val:.2f} ppm")
                print(f"    δ_yy: {delta_yy_val:.2f} ppm")
                print(f"    δ_zz: {delta_zz_val:.2f} ppm")
        
        # Evaluate Y_2^m using chosen method
        if self.use_sympy:
            # Method 1: Sympy symbolic calculation (lambdified functions)
            for m_idx, m in enumerate(range(-2, 3)):
                func = self.CSA_Y_lm_funcs[m_idx]
                Y2m_coefficients[:, m_idx] = func(alpha, beta, gamma, 
                                                  delta_xx_val, delta_yy_val, delta_zz_val)
            if self.config.verbose:
                print(f"  Using sympy-based symbolic calculation")
        else:
            # Method 2: NumPy optimized calculation (faster)
            Y2m_coefficients = self._transform_csa_tensor_numpy(alpha, beta, gamma,
                                                                delta_xx_val, delta_yy_val, delta_zz_val)
            if self.config.verbose:
                print(f"  Using NumPy-optimized calculation")
        
        # Take real part (imaginary parts should be numerically zero for physical systems)
        Y2m_coefficients = np.real(Y2m_coefficients)
        
        if self.config.verbose:
            print(f"\n  Y₂ₘ statistics (full Wigner D-matrix calculation):")
            for m in range(-2, 3):
                m_idx = m + 2
                mean_val = np.mean(np.abs(Y2m_coefficients[:, m_idx]))
                std_val = np.std(np.abs(Y2m_coefficients[:, m_idx]))
                print(f"    Y₂^{m:+d}: mean={mean_val:.3f}, std={std_val:.3f}")
        
        return Y2m_coefficients
    
    def _calculate_dipolar(self, euler_angles: np.ndarray) -> np.ndarray:
        """
        Calculate Y₂ₘ coefficients for dipolar coupling.
        
        Dipolar coupling is axially symmetric (η = 0) along internuclear vector.
        Only Y₂₀ component is non-zero.
        
        Dipolar coupling constant:
            D = -(μ₀/4π) × (γᵢγⱼℏ) / r³
        
        Parameters
        ----------
        euler_angles : np.ndarray
            Euler angles (n_steps, 3)
        
        Returns
        -------
        Y2m_coefficients : np.ndarray
            Y₂ₘ coefficients (n_steps, 5)
        """
        if self.config.verbose:
            print(f"  Dipolar coupling (axially symmetric)")
            print(f"  Only Y₂₀ component calculated")
        
        n_steps = euler_angles.shape[0]
        Y2m_coefficients = np.zeros((n_steps, 5), dtype=complex)
        
        # Extract beta (polar angle)
        beta = euler_angles[:, 1]
        
        # Dipolar coupling constant (placeholder)
        # User should provide or calculate based on nuclei and distance
        D_coupling = 10000.0  # Hz (typical for 15N-1H at 1.02 Å)
        
        # Y₂₀ component (axial)
        Y2m_coefficients[:, 2] = D_coupling * (3 * np.cos(beta)**2 - 1) / 2
        
        if self.config.verbose:
            print(f"  D coupling constant: {D_coupling:.1f} Hz")
            print(f"  Y₂₀ mean: {np.mean(np.abs(Y2m_coefficients[:, 2])):.2f}")
        
        return Y2m_coefficients
    
    @staticmethod
    def calculate_dipolar_coupling_constant(gamma_i: float, 
                                           gamma_j: float, 
                                           distance: float) -> float:
        """
        Calculate dipolar coupling constant.
        
        D = -(μ₀/4π) × (γᵢγⱼℏ) / r³
        
        Parameters
        ----------
        gamma_i : float
            Gyromagnetic ratio of nucleus i (rad/T)
        gamma_j : float
            Gyromagnetic ratio of nucleus j (rad/T)
        distance : float
            Internuclear distance (meters)
        
        Returns
        -------
        D : float
            Dipolar coupling constant (Hz)
        """
        # Physical constants
        mu_0 = 4 * np.pi * 1e-7  # Vacuum permeability (T²m³/J)
        hbar = 1.054571817e-34   # Reduced Planck's constant (J·s)
        
        # Dipolar coupling constant
        D = -(mu_0 / (4 * np.pi)) * (gamma_i * gamma_j * hbar) / (distance**3)
        
        # Convert to Hz
        D_Hz = D / (2 * np.pi)
        
        return D_Hz
    
    @staticmethod
    def CSA_tensor_to_parameters(sigma_11: float, 
                                 sigma_22: float, 
                                 sigma_33: float) -> Tuple[float, float, float]:
        """
        Convert CSA tensor principal values to δ_iso, Δδ, η.
        
        Convention: |σ_33 - σ_iso| ≥ |σ_11 - σ_iso| ≥ |σ_22 - σ_iso|
        
        Parameters
        ----------
        sigma_11, sigma_22, sigma_33 : float
            Principal components of CSA tensor (ppm)
        
        Returns
        -------
        delta_iso : float
            Isotropic chemical shift (ppm)
        delta_sigma : float
            Anisotropy Δδ = σ_33 - σ_iso (ppm)
        eta : float
            Asymmetry η = (σ_22 - σ_11) / Δδ
        """
        # Isotropic shift
        delta_iso = (sigma_11 + sigma_22 + sigma_33) / 3
        
        # Sort by distance from isotropic
        sigmas = np.array([sigma_11, sigma_22, sigma_33])
        sorted_sigmas = sigmas[np.argsort(np.abs(sigmas - delta_iso))]
        
        # Anisotropy (largest deviation)
        delta_sigma = sorted_sigmas[2] - delta_iso
        
        # Asymmetry
        eta = (sorted_sigmas[1] - sorted_sigmas[0]) / delta_sigma if delta_sigma != 0 else 0.0
        eta = np.clip(eta, 0, 1)  # Ensure 0 ≤ η ≤ 1
        
        return delta_iso, delta_sigma, eta
    
    @staticmethod
    def Y2m_to_cartesian(Y2m: np.ndarray, euler_angles: np.ndarray) -> np.ndarray:
        """
        Convert Y₂ₘ coefficients back to Cartesian tensor representation.
        
        Parameters
        ----------
        Y2m : np.ndarray
            Y₂ₘ coefficients (n_steps, 5)
        euler_angles : np.ndarray
            Euler angles (n_steps, 3)
        
        Returns
        -------
        cartesian_tensor : np.ndarray
            Cartesian tensor representation (n_steps, 3, 3)
        """
        n_steps = Y2m.shape[0]
        tensors = np.zeros((n_steps, 3, 3))
        
        # Inverse transformation from Y₂ₘ to Cartesian
        # (Detailed implementation requires Wigner-D matrices)
        
        # Placeholder for now
        return tensors
    
    def save(self, filepath: str):
        """
        Save Y₂ₘ coefficients.
        
        Parameters
        ----------
        filepath : str
            Output file path (.npz format)
        """
        if self.Y2m_coefficients is None:
            raise ValueError("No Y₂ₘ coefficients to save. Run calculate() first.")
        
        np.savez(filepath,
                Y2m_real=self.Y2m_coefficients.real,
                Y2m_imag=self.Y2m_coefficients.imag,
                config=str(self.config.to_dict()))
        
        if self.config.verbose:
            print(f"  ✓ Saved Y₂ₘ coefficients to: {filepath}")


# Example usage
if __name__ == '__main__':
    from .config import NMRConfig
    from .xyz_generator import TrajectoryGenerator
    from .euler_converter import EulerConverter
    
    # Test CSA
    config = NMRConfig(
        trajectory_type='diffusion_cone',
        S2=0.85,
        tau_c=2e-9,
        num_steps=100,
        interaction_type='CSA',
        delta_sigma=100.0,
        eta=0.3,
        verbose=True
    )
    
    # Generate trajectory
    gen = TrajectoryGenerator(config)
    rotations, _ = gen.generate()
    
    # Convert to Euler angles
    converter = EulerConverter(config)
    euler_angles = converter.convert(rotations=rotations)
    
    # Calculate Y₂ₘ
    sh_calc = SphericalHarmonicsCalculator(config)
    Y2m = sh_calc.calculate(euler_angles)
    
    print(f"\nY₂ₘ shape: {Y2m.shape}")
    print(f"Y₂₀ mean: {np.mean(np.abs(Y2m[:, 2])):.2f}")
