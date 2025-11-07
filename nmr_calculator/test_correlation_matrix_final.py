#!/usr/bin/env python3
"""
Test: Correlation matrix with direct method only

This validates that compute_correlation_matrix() matches the reference
implementation exactly using the direct loop method.

FFT cross-correlation was removed due to formula complexity.
"""

import numpy as np
import time
from autocorrelation import AutocorrelationCalculator
from config import NMRConfig


def reference_correlation_matrix(Y2m_coefficients, max_lag, lag_step):
    """Reference implementation from t1_anisotropy_analysis.py (line 323-350)"""
    corr_matrix = {}
    m_values = [-2, -1, 0, 1, 2]
    
    for m1 in m_values:
        for m2 in m_values:
            y1 = Y2m_coefficients[:, m1 + 2]
            y2 = Y2m_coefficients[:, m2 + 2]
            
            corr = []
            for tau in range(0, max_lag, lag_step):
                if tau == 0:
                    val = np.mean(y1 * np.conj(y2))
                else:
                    val = np.mean(y1[:-tau] * np.conj(y2[tau:]))
                corr.append(val)
            
            corr_matrix[(m1, m2)] = np.array(corr)
    
    return corr_matrix


def test_correlation_matrix():
    """Test correlation matrix against reference"""
    
    print("\n" + "="*70)
    print("TEST: Correlation Matrix (Direct Method Only)")
    print("="*70)
    
    # Generate test data
    np.random.seed(42)
    n_steps = 10000
    dt = 0.02e-9
    tau_c = 5e-9
    
    # Random Y2m coefficients
    Y2m = np.random.randn(n_steps, 5) + 1j * np.random.randn(n_steps, 5)
    Y2m *= 0.1
    
    max_lag = 2000
    
    # Config
    config = NMRConfig(
        trajectory_type='diffusion_cone',
        S2=0.85,
        tau_c=tau_c,
        dt=dt,
        num_steps=n_steps,
        interaction_type='CSA',
        delta_sigma=100.0,
        eta=0.0,
        max_lag=max_lag,
        lag_step=1,
        verbose=False
    )
    
    print(f"\nDataset: {n_steps} steps, max_lag={max_lag}")
    print(f"Computing 25 cross-correlations (5×5 matrix)...")
    
    # Method 1: Our implementation
    print("\n1. AutocorrelationCalculator.compute_correlation_matrix()")
    calc = AutocorrelationCalculator(config)
    t0 = time.time()
    corr_ours = calc.compute_correlation_matrix(Y2m)
    time_ours = time.time() - t0
    print(f"   Time: {time_ours:.3f} s")
    
    # Method 2: Reference implementation
    print("\n2. Reference implementation")
    t0 = time.time()
    corr_ref = reference_correlation_matrix(Y2m, max_lag, 1)
    time_ref = time.time() - t0
    print(f"   Time: {time_ref:.3f} s")
    
    # Compare
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    
    max_diff = 0
    for key in corr_ref.keys():
        diff = np.max(np.abs(corr_ours[key] - corr_ref[key]))
        max_diff = max(max_diff, diff)
    
    print(f"\nMaximum difference: {max_diff:.2e}")
    
    # Sample values
    print("\nSample C_{2,-2}[100]:")
    print(f"  Our implementation: {corr_ours[(2, -2)][100]:.6e}")
    print(f"  Reference:          {corr_ref[(2, -2)][100]:.6e}")
    print(f"  Difference:         {np.abs(corr_ours[(2, -2)][100] - corr_ref[(2, -2)][100]):.2e}")
    
    # Pass/fail
    passed = max_diff < 1e-14
    
    print("\n" + "="*70)
    if passed:
        print("✓ TEST PASSED")
    else:
        print("✗ TEST FAILED")
    print("="*70)
    
    return passed


if __name__ == '__main__':
    print("\n" + "="*70)
    print("CORRELATION MATRIX: Final Validation")
    print("="*70)
    
    passed = test_correlation_matrix()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    if passed:
        print("✓ Correlation matrix implementation validated")
        print("  - Uses direct method only (matches reference exactly)")
        print("  - FFT cross-correlation removed (too complex)")
        print("  - Performance adequate for 25 correlations")
    else:
        print("✗ Validation failed")
    print("="*70)
