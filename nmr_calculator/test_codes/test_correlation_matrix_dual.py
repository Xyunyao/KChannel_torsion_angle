"""
Test correlation matrix with both Direct and FFT methods.
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import NMRConfig
from autocorrelation import AutocorrelationCalculator


def reference_correlation_matrix(Y_series, max_lag, lag_step):
    """Reference from t1_anisotropy_analysis.py"""
    corr_matrix = {}
    for m1 in range(-2, 3):
        for m2 in range(-2, 3):
            corr = []
            y1 = Y_series[:, m1 + 2]
            y2 = Y_series[:, m2 + 2]
            for tau in range(0, max_lag, lag_step):
                val = np.mean(y1[:-tau or None] * np.conj(y2[tau:])) if tau > 0 else np.mean(y1 * np.conj(y2))
                corr.append(val)
            corr_matrix[(m1, m2)] = np.array(corr)
    return corr_matrix


def test_correlation_matrix_methods():
    """Test that both methods give identical results for correlation matrix."""
    
    print("="*70)
    print("TEST: Correlation Matrix - Direct vs FFT")
    print("="*70)
    
    # Create test Y2m
    np.random.seed(42)
    n_steps = 10000
    tau_c = 2e-9
    dt = 1e-12
    
    t = np.arange(n_steps) * dt
    Y2m = np.zeros((n_steps, 5), dtype=complex)
    for m_idx in range(5):
        # Each m has different dynamics
        Y2m[:, m_idx] = np.exp(-t / (tau_c * (1 + 0.2 * m_idx))) * (1 + 0.1 * np.random.randn(n_steps))
    
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
    
    # Method 1: Direct (reference)
    print("\n1. Direct method (reference)")
    calc_direct = AutocorrelationCalculator(config, use_fft=False)
    t0 = time.time()
    corr_direct = calc_direct.compute_correlation_matrix(Y2m, use_fft=False)
    time_direct = time.time() - t0
    print(f"   Time: {time_direct:.3f} s")
    
    # Method 2: FFT
    print("\n2. FFT method (fast)")
    calc_fft = AutocorrelationCalculator(config, use_fft=True)
    t0 = time.time()
    corr_fft = calc_fft.compute_correlation_matrix(Y2m, use_fft=True)
    time_fft = time.time() - t0
    print(f"   Time: {time_fft:.3f} s")
    
    # Method 3: Reference implementation
    print("\n3. Reference implementation")
    t0 = time.time()
    corr_ref = reference_correlation_matrix(Y2m, max_lag, 1)
    time_ref = time.time() - t0
    print(f"   Time: {time_ref:.3f} s")
    
    # Compare all methods
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    
    max_diff_direct_ref = 0
    max_diff_fft_ref = 0
    max_diff_direct_fft = 0
    
    for m1 in range(-2, 3):
        for m2 in range(-2, 3):
            diff_dr = np.max(np.abs(corr_direct[(m1, m2)] - corr_ref[(m1, m2)]))
            diff_fr = np.max(np.abs(corr_fft[(m1, m2)] - corr_ref[(m1, m2)]))
            diff_df = np.max(np.abs(corr_direct[(m1, m2)] - corr_fft[(m1, m2)]))
            
            if diff_dr > max_diff_direct_ref:
                max_diff_direct_ref = diff_dr
            if diff_fr > max_diff_fft_ref:
                max_diff_fft_ref = diff_fr
            if diff_df > max_diff_direct_fft:
                max_diff_direct_fft = diff_df
    
    print(f"\nMax differences:")
    print(f"  Direct vs Reference:    {max_diff_direct_ref:.6e}")
    print(f"  FFT vs Reference:       {max_diff_fft_ref:.6e}")
    print(f"  Direct vs FFT:          {max_diff_direct_fft:.6e}")
    
    print(f"\nPerformance:")
    print(f"  Direct: {time_direct:.3f} s")
    print(f"  FFT:    {time_fft:.3f} s")
    print(f"  Speedup: {time_direct/time_fft:.1f}×")
    
    # Sample values
    print(f"\nSample correlations:")
    print(f"  C_{{0,0}}[0]:")
    print(f"    Direct: {corr_direct[(0,0)][0]:.6e}")
    print(f"    FFT:    {corr_fft[(0,0)][0]:.6e}")
    print(f"    Ref:    {corr_ref[(0,0)][0]:.6e}")
    
    print(f"  C_{{0,1}}[0]:")
    print(f"    Direct: {corr_direct[(0,1)][0]:.6e}")
    print(f"    FFT:    {corr_fft[(0,1)][0]:.6e}")
    print(f"    Ref:    {corr_ref[(0,1)][0]:.6e}")
    
    print(f"  C_{{2,-2}}[100]:")
    print(f"    Direct: {corr_direct[(2,-2)][100]:.6e}")
    print(f"    FFT:    {corr_fft[(2,-2)][100]:.6e}")
    print(f"    Ref:    {corr_ref[(2,-2)][100]:.6e}")
    
    # Pass/fail
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    tol = 1e-10
    all_pass = True
    
    if max_diff_direct_ref < tol:
        print("✓ Direct method matches reference")
    else:
        print(f"✗ Direct method differs: {max_diff_direct_ref:.2e}")
        all_pass = False
    
    if max_diff_fft_ref < tol:
        print("✓ FFT method matches reference")
    else:
        print(f"✗ FFT method differs: {max_diff_fft_ref:.2e}")
        all_pass = False
    
    if max_diff_direct_fft < tol:
        print("✓ Direct and FFT methods are identical")
    else:
        print(f"✗ Direct and FFT differ: {max_diff_direct_fft:.2e}")
        all_pass = False
    
    return all_pass


def test_large_dataset():
    """Test performance on large dataset."""
    
    print("\n" + "="*70)
    print("TEST: Large Dataset Performance")
    print("="*70)
    
    # Large dataset
    np.random.seed(123)
    n_steps = 50000
    Y2m = np.random.randn(n_steps, 5) + 1j * np.random.randn(n_steps, 5)
    
    max_lag = 5000
    
    config = NMRConfig(
        trajectory_type='diffusion_cone',
        S2=0.85,
        tau_c=2e-9,
        dt=1e-12,
        num_steps=n_steps,
        interaction_type='CSA',
        delta_sigma=100.0,
        eta=0.0,
        max_lag=max_lag,
        lag_step=1,
        verbose=False
    )
    
    print(f"\nDataset: {n_steps} steps, max_lag={max_lag}")
    print("Computing 25 cross-correlations...")
    
    # Direct method
    calc_direct = AutocorrelationCalculator(config, use_fft=False)
    t0 = time.time()
    corr_direct = calc_direct.compute_correlation_matrix(Y2m, use_fft=False)
    time_direct = time.time() - t0
    
    # FFT method
    calc_fft = AutocorrelationCalculator(config, use_fft=True)
    t0 = time.time()
    corr_fft = calc_fft.compute_correlation_matrix(Y2m, use_fft=True)
    time_fft = time.time() - t0
    
    # Compare
    max_diff = 0
    for m1 in range(-2, 3):
        for m2 in range(-2, 3):
            diff = np.max(np.abs(corr_direct[(m1, m2)] - corr_fft[(m1, m2)]))
            if diff > max_diff:
                max_diff = diff
    
    print(f"\nPerformance:")
    print(f"  Direct: {time_direct:.2f} s")
    print(f"  FFT:    {time_fft:.2f} s")
    print(f"  Speedup: {time_direct/time_fft:.1f}×")
    
    print(f"\nAccuracy:")
    print(f"  Max difference: {max_diff:.6e}")
    
    if max_diff < 1e-10:
        print("  ✓ Methods are identical")
        return True
    else:
        print(f"  ✗ Methods differ by {max_diff:.2e}")
        return False


if __name__ == '__main__':
    print("\n" + "="*70)
    print("CORRELATION MATRIX: Dual Method Validation")
    print("="*70)
    
    results = []
    results.append(test_correlation_matrix_methods())
    results.append(test_large_dataset())
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Tests passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("\n✓ ALL TESTS PASSED!")
        print("\nCorrelation matrix supports both methods:")
        print("  1. Direct (default): Exact, matches reference")
        print("  2. FFT (use_fft=True): Much faster for large datasets")
        print("\nUsage:")
        print("  # Use instance setting")
        print("  calc = AutocorrelationCalculator(config, use_fft=True)")
        print("  corr_matrix = calc.compute_correlation_matrix(Y2m)")
        print("  ")
        print("  # Or override per call")
        print("  corr_matrix = calc.compute_correlation_matrix(Y2m, use_fft=False)")
    else:
        print("\n✗ SOME TESTS FAILED")
    
    print("="*70)
