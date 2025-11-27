"""
Module: Analytical Ensemble Average (OPTIMIZED)

Optimized calculation of ensemble average using Wigner D orthogonality.

Key insight: Due to Wigner D orthogonality over SO(3):
∫ D_{mj}(Ω) · D*_{mn}(Ω) dΩ = δ_{jn}/(2ℓ+1) = δ_{jn}/5  (for ℓ=2)

Therefore:
<C_rot(m,m,t)> = Σ_{j,n} K_{jn} · C_{jn}(t)
               = Σ_j (1/5) · C_{jj}(t)
               = (1/5) · Tr[C(t)]

This is the isotropic average - just the trace of the correlation matrix divided by 5!
"""

import numpy as np
from typing import Dict, Tuple
from .config import NMRConfig


class AnalyticalEnsembleCalculator:
    """
    Calculate analytical ensemble average using Wigner D orthogonality.
    
    Key insight: Due to orthogonality, ensemble averaging over SO(3) gives:
    K_{jn} = δ_{jn}/5 (Kronecker delta)
    
    So the ensemble average is simply: <C_rot(m,m,t)> = (1/5) · Tr[C(t)]
    
    Attributes
    ----------
    config : NMRConfig
        Configuration object
    K_matrix : Dict[Tuple[int,int], float]
        Weight coefficients K_jn (diagonal = 1/5, off-diagonal = 0)
    current_target_m : int
        Current target m value
    """
    
    def __init__(self, config: NMRConfig):
        """
        Initialize analytical ensemble calculator.
        
        Parameters
        ----------
        config : NMRConfig
            Configuration object
        """
        self.config = config
        
        # K-matrix is diagonal with K_jj = 1/5 due to orthogonality
        self.K_matrix = self._build_K_matrix()
        self.current_target_m = None
        
        if self.config.verbose:
            print(f"\n{'='*70}")
            print("Analytical Ensemble Average Calculator (OPTIMIZED)")
            print(f"{'='*70}")
            print(f"  Using Wigner D orthogonality: K_jn = δ_jn/5")
            print(f"  Result: <C_rot(m,m,t)> = (1/5) · Σ_j C_jj(t)")
    
    def _build_K_matrix(self) -> Dict[Tuple[int, int], float]:
        """
        Build K-matrix using orthogonality relation.
        
        Due to Wigner D orthogonality:
        ∫ D_{mj}(Ω) · D*_{mn}(Ω) dΩ = δ_{jn}/(2ℓ+1) = δ_{jn}/5  (for ℓ=2)
        
        Returns
        -------
        K_matrix : Dict[Tuple[int,int], float]
            K_jn coefficients (diagonal = 1/5, off-diagonal = 0)
        """
        K_matrix = {}
        
        for j in [-2, -1, 0, 1, 2]:
            for n in [-2, -1, 0, 1, 2]:
                if j == n:
                    K_matrix[(j, n)] = 1.0 / 5.0  # Diagonal elements
                else:
                    K_matrix[(j, n)] = 0.0  # Off-diagonal elements
        
        return K_matrix
    
    def integrate_weight_matrix(self, target_m: int = 1) -> Dict[Tuple[int, int], float]:
        """
        Return pre-computed K-matrix based on orthogonality.
        
        No integration needed! Wigner D orthogonality gives:
        K_jn = δ_jn/5
        
        Parameters
        ----------
        target_m : int
            Target m value (not used, kept for API compatibility)
        
        Returns
        -------
        K_matrix : Dict[Tuple[int,int], float]
            K_jn coefficients (diagonal = 1/5, off-diagonal = 0)
        """
        self.current_target_m = target_m
        
        if self.config.verbose:
            print(f"\n  Using analytical K-matrix from Wigner D orthogonality:")
            print(f"    K_jn = δ_jn/5  (Kronecker delta)")
            print(f"    Non-zero coefficients: 5/25 (diagonal only)")
        
        return self.K_matrix
    
    def evaluate_ensemble_average(self, 
                                  C_matrix: Dict[Tuple[int, int], np.ndarray],
                                  target_m: int = 1) -> np.ndarray:
        """
        Evaluate ensemble average using orthogonality relation.
        
        <C_rot(m,m,t)> = Σ_jn K_jn · C_jn(t)
                       = Σ_j (1/5) · C_jj(t)
                       = (1/5) · Tr[C(t)]
        
        This is extremely fast - just sum the diagonal elements!
        
        Parameters
        ----------
        C_matrix : Dict[Tuple[int,int], np.ndarray]
            Correlation matrix C(j,n,t) from simulation
        target_m : int
            Target m value (not used, all diagonal elements are equivalent)
        
        Returns
        -------
        ensemble_avg : np.ndarray
            Ensemble-averaged correlation function
        """
        if self.current_target_m != target_m:
            self.current_target_m = target_m
        
        if self.config.verbose:
            print(f"\n  Evaluating <C_rot({target_m},{target_m},t)> using orthogonality...")
            print(f"    Formula: <C_rot> = (1/5) · Σ_j C_jj(t)")
        
        # Get time array length
        n_time = len(C_matrix[(0, 0)])
        result = np.zeros(n_time, dtype=complex)
        
        # Sum diagonal elements: Σ_j C_jj(t)
        for j in [-2, -1, 0, 1, 2]:
            result += C_matrix[(j, j)]
        
        # Multiply by 1/5
        result /= 5.0
        
        if self.config.verbose:
            print(f"    ✓ Evaluated for {n_time} time points")
            print(f"    Initial value: {result[0].real:.6f}")
            print(f"    Final value: {result[-1].real:.6f}")
        
        return result.real


# Test the optimized implementation
if __name__ == '__main__':
    from .config import NMRConfig
    
    print("="*70)
    print("Testing Optimized Analytical Ensemble Calculator")
    print("="*70)
    
    config = NMRConfig(verbose=True)
    calc = AnalyticalEnsembleCalculator(config)
    
    # Show K-matrix
    print("\n" + "="*70)
    print("K-Matrix (from Wigner D orthogonality)")
    print("="*70)
    
    K_matrix = calc.integrate_weight_matrix(target_m=1)
    
    print("\nK_jn values:")
    print("  Diagonal: K_jj = 1/5 = 0.20000")
    print("  Off-diagonal: K_jn = 0 (j ≠ n)")
    
    print("\n" + "="*70)
    print("✓ K-matrix is trivial due to orthogonality!")
    print("  Ensemble average = (1/5) × sum of diagonal C_jj(t)")
    print("  No expensive symbolic integration needed!")
    print("="*70)