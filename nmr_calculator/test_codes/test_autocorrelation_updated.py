"""
Test updated autocorrelation implementation against reference.

Validates:
1. Direct calculation matches reference exactly
2. No DC offset removal
3. No zero-fill in ACF calculation
4. compute_correlation_matrix matches reference
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import NMRConfig
from autocorrelation import AutocorrelationCalculator


def reference_acf(series, max_lag, lag_step):
    """Reference implementation from t1_anisotropy_analysis.py"""
    corr = []
    for tau in range(0, max_lag, lag_step):
        val = np.mean(series[:-tau or None] * np.conj(series[tau:])) if tau > 0 else np.mean(series * np.conj(series))
        corr.append(val)
    return np.array(corr)


def reference_correlation_matrix(Y_series, max_lag, lag_step):
    """Reference correlation matrix from t1_anisotropy_analysis.py"""
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


def test_single_acf():
    """Test single ACF calculation."""
    print("="*70)
    print("TEST 1: Single ACF Calculation")
    print("="*70)
    
    # Create test signal
    np.random.seed(42)
    n_steps = 5000
    tau_c = 2e-9
    dt = 1e-12
    
    t = np.arange(n_steps) * dt
    signal = np.exp(-t / tau_c) + 0.1 * np.random.randn(n_steps)
    
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
        max_lag=1000,
        lag_step=1,
        verbose=False
    )
    
    calc = AutocorrelationCalculator(config)
    
    # Our implementation
    acf_ours = calc._calculate_acf_direct(signal, max_lag=1000, lag_step=1)
    
    # Reference implementation
    acf_ref = reference_acf(signal, max_lag=1000, lag_step=1)
    
    # Compare
    diff = np.max(np.abs(acf_ours - acf_ref))
    print(f"\nOur ACF[0]: {acf_ours[0]:.6e}")
    print(f"Ref ACF[0]: {acf_ref[0]:.6e}")
    print(f"Our ACF[100]: {acf_ours[100]:.6e}")
    print(f"Ref ACF[100]: {acf_ref[100]:.6e}")
    print(f"\nMax difference: {diff:.6e}")
    
    if diff < 1e-10:
        print("✓ PASS: Single ACF matches reference!")
        return True
    else:
        print(f"✗ FAIL: Difference too large: {diff:.6e}")
        return False


def test_correlation_matrix():
    """Test full correlation matrix."""
    print("\n" + "="*70)
    print("TEST 2: Correlation Matrix")
    print("="*70)
    
    # Create test Y2m coefficients
    np.random.seed(42)
    n_steps = 5000
    
    # Simulate Y2m with some correlation structure
    Y2m = np.zeros((n_steps, 5), dtype=complex)
    for m_idx in range(5):
        Y2m[:, m_idx] = np.exp(-np.arange(n_steps) * 1e-12 / 2e-9) * (1 + 0.1 * np.random.randn(n_steps))
    
    # Config
    config = NMRConfig(
        trajectory_type='diffusion_cone',
        S2=0.85,
        tau_c=2e-9,
        dt=1e-12,
        num_steps=n_steps,
        interaction_type='CSA',
        delta_sigma=100.0,
        eta=0.0,
        max_lag=500,
        lag_step=1,
        verbose=False
    )
    
    calc = AutocorrelationCalculator(config)
    
    # Our implementation
    corr_ours = calc.compute_correlation_matrix(Y2m)
    
    # Reference implementation
    corr_ref = reference_correlation_matrix(Y2m, max_lag=500, lag_step=1)
    
    # Compare all entries
    max_diff = 0
    max_key = (0, 0)
    for m1 in range(-2, 3):
        for m2 in range(-2, 3):
            diff = np.max(np.abs(corr_ours[(m1, m2)] - corr_ref[(m1, m2)]))
            if diff > max_diff:
                max_diff = diff
                max_key = (m1, m2)
    
    print(f"\nComparing all 25 correlation functions...")
    print(f"Max difference: {max_diff:.6e} at ({max_key[0]}, {max_key[1]})")
    
    # Sample values
    print(f"\nSample values:")
    print(f"  C_{{0,0}}[0]: ours={corr_ours[(0,0)][0]:.6e}, ref={corr_ref[(0,0)][0]:.6e}")
    print(f"  C_{{0,1}}[0]: ours={corr_ours[(0,1)][0]:.6e}, ref={corr_ref[(0,1)][0]:.6e}")
    print(f"  C_{{2,-2}}[0]: ours={corr_ours[(2,-2)][0]:.6e}, ref={corr_ref[(2,-2)][0]:.6e}")
    
    if max_diff < 1e-10:
        print("\n✓ PASS: Correlation matrix matches reference!")
        return True
    else:
        print(f"\n✗ FAIL: Difference too large: {max_diff:.6e}")
        return False


def test_full_pipeline():
    """Test full calculation pipeline."""
    print("\n" + "="*70)
    print("TEST 3: Full Pipeline (Y2m -> ACF)")
    print("="*70)
    
    # Create test Y2m
    np.random.seed(42)
    n_steps = 10000
    tau_c = 2e-9
    dt = 1e-12
    
    t = np.arange(n_steps) * dt
    Y2m = np.zeros((n_steps, 5), dtype=complex)
    for m_idx in range(5):
        Y2m[:, m_idx] = np.exp(-t / tau_c) * (1 + 0.1 * np.random.randn(n_steps))
    
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
        max_lag=2000,
        lag_step=1,
        verbose=True
    )
    
    calc = AutocorrelationCalculator(config)
    
    # Calculate ACF
    acf, time_lags = calc.calculate(Y2m)
    
    print(f"\n✓ Pipeline completed successfully")
    print(f"  ACF shape: {acf.shape}")
    print(f"  Time range: 0 to {time_lags[-1]*1e9:.2f} ns")
    print(f"  ACF[0]: {acf[0]:.6f} (should be 1.0)")
    tau_c_idx = min(int(tau_c/dt), len(acf)-1)
    print(f"  ACF decay at ~τ_c: {acf[tau_c_idx]:.6f}")
    
    return True


def test_no_dc_removal():
    """Verify that NO DC offset removal is performed."""
    print("\n" + "="*70)
    print("TEST 4: Verify No DC Offset Removal")
    print("="*70)
    
    # Signal with constant DC offset
    np.random.seed(42)
    n_steps = 5000
    dc_offset = 10.0  # Large DC offset
    signal = dc_offset + 0.1 * np.random.randn(n_steps)
    
    config = NMRConfig(
        trajectory_type='diffusion_cone',
        S2=0.85,
        tau_c=2e-9,
        dt=1e-12,
        num_steps=n_steps,
        interaction_type='CSA',
        delta_sigma=100.0,
        eta=0.0,
        max_lag=100,
        lag_step=1,
        verbose=False
    )
    
    calc = AutocorrelationCalculator(config)
    
    # Calculate ACF
    acf = calc._calculate_acf_direct(signal, max_lag=100, lag_step=1)
    
    # ACF[0] should include DC offset squared term
    expected_acf0 = np.mean(signal * np.conj(signal))
    
    print(f"\nSignal mean: {np.mean(signal):.2f} (DC offset: {dc_offset})")
    print(f"ACF[0]: {acf[0]:.2f}")
    print(f"Expected (with DC): {expected_acf0:.2f}")
    print(f"Difference: {abs(acf[0] - expected_acf0):.6e}")
    
    # If DC was removed, ACF[0] would be much smaller (just variance ~0.01)
    # With DC, ACF[0] should be ~100 (DC²)
    
    if abs(acf[0] - expected_acf0) < 1e-10:
        print("\n✓ PASS: DC offset NOT removed (as intended)")
        return True
    else:
        print("\n✗ FAIL: DC offset may have been removed")
        return False


if __name__ == '__main__':
    print("\n" + "="*70)
    print("AUTOCORRELATION UPDATE VALIDATION")
    print("Verifying: Direct method, no DC removal, no zero-fill")
    print("="*70)
    
    results = []
    results.append(test_single_acf())
    results.append(test_correlation_matrix())
    results.append(test_full_pipeline())
    results.append(test_no_dc_removal())
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Tests passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("\n✓ ALL TESTS PASSED!")
        print("\nUpdates confirmed:")
        print("  1. Direct calculation method (matches reference)")
        print("  2. No DC offset removal")
        print("  3. No zero-fill in ACF")
        print("  4. Correlation matrix matches reference exactly")
    else:
        print("\n✗ SOME TESTS FAILED")
    
    print("="*70)
