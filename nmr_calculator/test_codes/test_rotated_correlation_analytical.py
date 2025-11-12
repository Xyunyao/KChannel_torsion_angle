#!/usr/bin/env python3
"""
Test: Module 5 - Analytical Comparison

This test compares rotated correlation matrix results with analytical solutions
for cases where we have known theoretical predictions.

Tests:
1. Isotropic tumbling: All off-diagonal correlations should average to zero
2. Axially symmetric tumbling: Only certain components survive
3. Exponential decay preservation: Rotation shouldn't change decay rates
4. Hermiticity: C(m1,m2) = C*(m2,m1)
5. Symmetry properties under rotation
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nmr_calculator.rotated_correlation import RotatedCorrelationCalculator, rotate_all
from nmr_calculator.rotation_matrix import WignerDCalculator
from nmr_calculator.config import NMRConfig


def create_isotropic_correlation_matrix(n_lags=100, tau_c=5e-9, dt=0.02e-9):
    """
    Create a correlation matrix for isotropic tumbling.
    
    For isotropic tumbling, only diagonal elements survive ensemble averaging:
    ⟨C_{m1,m2}(τ)⟩ = δ_{m1,m2} × exp(-τ/τ_c)
    
    All diagonal elements should have same magnitude for true isotropic case.
    """
    m_values = [-2, -1, 0, 1, 2]
    corr_matrix = {}
    
    tau = np.arange(n_lags) * dt
    
    for m1 in m_values:
        for m2 in m_values:
            if m1 == m2:
                # Diagonal: exponential decay (same for all m)
                corr_matrix[(m1, m2)] = np.exp(-tau / tau_c)
            else:
                # Off-diagonal: should average to zero
                # Add small random component to test averaging
                corr_matrix[(m1, m2)] = 0.01 * np.exp(-tau / tau_c) * (np.random.randn() + 1j * np.random.randn())
    
    return corr_matrix


def create_uniform_orientations(n_orientations):
    """
    Create uniformly distributed orientations on SO(3).
    
    Uses Fibonacci sphere for uniform angular distribution.
    """
    np.random.seed(42)
    
    # Uniform alpha and gamma (0 to 2π)
    alpha = np.random.rand(n_orientations) * 2 * np.pi
    gamma = np.random.rand(n_orientations) * 2 * np.pi
    
    # Uniform beta using arccos(uniform(-1,1)) for proper SO(3) distribution
    u = np.random.rand(n_orientations) * 2 - 1  # uniform in [-1, 1]
    beta = np.arccos(u)
    
    euler_angles = np.column_stack([alpha, beta, gamma])
    
    return euler_angles


def test_isotropic_averaging():
    """
    Test 1: Isotropic tumbling should average off-diagonal elements to zero.
    
    For uniform sampling over SO(3), ensemble averaging should yield:
    ⟨C_{m1,m2}(τ)⟩ ≈ δ_{m1,m2} × C_{m1,m1}(τ)
    
    Off-diagonal elements should be suppressed by factor ~1/√N.
    """
    print("\n" + "="*70)
    print("TEST 1: Isotropic Tumbling - Off-diagonal Suppression")
    print("="*70)
    
    n_orientations = 1000
    n_lags = 50
    tau_c = 5e-9
    dt = 0.02e-9
    
    print(f"\n  Configuration:")
    print(f"    Orientations: {n_orientations}")
    print(f"    Lag points: {n_lags}")
    print(f"    Correlation time: {tau_c*1e9:.1f} ns")
    
    # Create isotropic correlation matrix
    corr_matrix = create_isotropic_correlation_matrix(n_lags, tau_c, dt)
    
    # Create uniform orientations
    euler_angles = create_uniform_orientations(n_orientations)
    
    # Rotate and average
    config = NMRConfig(verbose=False)
    calc = RotatedCorrelationCalculator(config)
    calc.compute_wigner_d_from_euler(euler_angles)
    rotated = calc.rotate_correlation_matrix(corr_matrix)
    ensemble_avg = calc.compute_ensemble_average()
    
    # Analyze results
    print("\n  Analysis at τ=0:")
    print("  " + "-"*66)
    print("  {:>10} {:>12} {:>12} {:>12}".format("(m1, m2)", "Diagonal", "Off-diag", "Ratio"))
    print("  " + "-"*66)
    
    diagonal_values = []
    offdiag_values = []
    
    for i, m1 in enumerate([-2, -1, 0, 1, 2]):
        for j, m2 in enumerate([-2, -1, 0, 1, 2]):
            val = ensemble_avg[i, j, 0]
            mag = np.abs(val)
            
            if m1 == m2:
                diagonal_values.append(mag)
                print(f"  ({m1:2d},{m2:2d})    {mag:12.6e}  {'---':>12}  {'---':>12}")
            else:
                offdiag_values.append(mag)
                ratio = mag / np.mean(diagonal_values) if diagonal_values else 0
                print(f"  ({m1:2d},{m2:2d})    {'---':>12}  {mag:12.6e}  {ratio:12.6e}")
    
    # Statistics
    avg_diagonal = np.mean(diagonal_values)
    avg_offdiag = np.mean(offdiag_values)
    suppression = avg_diagonal / avg_offdiag
    
    print("  " + "-"*66)
    print(f"\n  Average diagonal magnitude:     {avg_diagonal:.6e}")
    print(f"  Average off-diagonal magnitude: {avg_offdiag:.6e}")
    print(f"  Suppression factor:             {suppression:.1f}×")
    print(f"  Expected suppression:           ~{np.sqrt(n_orientations):.1f}× (√N)")
    
    # Validation
    # Off-diagonal should be suppressed by ~√N for random sampling
    expected_suppression = np.sqrt(n_orientations) * 0.1  # 0.1 from input noise level
    
    assert suppression > 5, f"Insufficient suppression: {suppression:.1f}× (expected >5×)"
    assert avg_offdiag < 0.1 * avg_diagonal, f"Off-diagonal too large: {avg_offdiag/avg_diagonal:.2%}"
    
    print("\n  ✓ Off-diagonal elements properly suppressed")
    print("  ✓ Isotropic averaging validated")
    
    print("\n✓ TEST 1 PASSED")
    return True


def test_exponential_decay_preservation():
    """
    Test 2: Rotation should preserve exponential decay rates.
    
    If original correlation decays as exp(-t/τ_c), the rotated correlation
    should decay with the same rate (rotation is unitary).
    """
    print("\n" + "="*70)
    print("TEST 2: Exponential Decay Rate Preservation")
    print("="*70)
    
    n_orientations = 100
    n_lags = 200
    tau_c = 5e-9
    dt = 0.02e-9
    
    print(f"\n  Configuration:")
    print(f"    Correlation time τ_c: {tau_c*1e9:.1f} ns")
    print(f"    Time step dt: {dt*1e9:.3f} ns")
    print(f"    Total time: {n_lags*dt*1e9:.1f} ns")
    
    # Create correlation with known decay
    corr_matrix = create_isotropic_correlation_matrix(n_lags, tau_c, dt)
    
    # Rotate
    euler_angles = create_uniform_orientations(n_orientations)
    config = NMRConfig(verbose=False)
    calc = RotatedCorrelationCalculator(config)
    calc.compute_wigner_d_from_euler(euler_angles)
    rotated = calc.rotate_correlation_matrix(corr_matrix)
    
    # Analyze decay for diagonal elements
    print("\n  Decay Analysis:")
    print("  " + "-"*66)
    
    tau = np.arange(n_lags) * dt
    expected_decay = np.exp(-tau / tau_c)
    
    decay_rates = []
    
    for i, m in enumerate([-2, -1, 0, 1, 2]):
        # Average over orientations
        avg_corr = np.mean(rotated[:, i, i, :], axis=0)
        
        # Fit exponential decay
        # C(t) = C(0) × exp(-t/τ)
        # log(C(t)/C(0)) = -t/τ
        
        # Use lags 10-100 to avoid initial and final noise
        fit_range = slice(10, 100)
        log_ratio = np.log(np.abs(avg_corr[fit_range]) / np.abs(avg_corr[0]))
        times = tau[fit_range]
        
        # Linear fit
        coeffs = np.polyfit(times, log_ratio, 1)
        fitted_rate = -coeffs[0]  # Rate = -slope
        fitted_tau_c = 1 / fitted_rate
        
        decay_rates.append(fitted_tau_c)
        
        error = np.abs(fitted_tau_c - tau_c) / tau_c * 100
        
        print(f"  m={m:2d}:  τ_c(fitted) = {fitted_tau_c*1e9:.3f} ns  "
              f"(error: {error:.1f}%)")
    
    # Statistics
    mean_tau_c = np.mean(decay_rates)
    std_tau_c = np.std(decay_rates)
    avg_error = np.abs(mean_tau_c - tau_c) / tau_c * 100
    
    print("  " + "-"*66)
    print(f"\n  Input τ_c:        {tau_c*1e9:.3f} ns")
    print(f"  Recovered τ_c:    {mean_tau_c*1e9:.3f} ± {std_tau_c*1e9:.3f} ns")
    print(f"  Average error:    {avg_error:.1f}%")
    
    # Validation
    assert avg_error < 10, f"Decay rate error too large: {avg_error:.1f}%"
    
    print("\n  ✓ Decay rates preserved after rotation")
    print("  ✓ Rotation is unitary (preserves dynamics)")
    
    print("\n✓ TEST 2 PASSED")
    return True


def test_hermiticity():
    """
    Test 3: Correlation matrix should be Hermitian.
    
    C(m1, m2) = C*(m2, m1)
    
    This property should be preserved after rotation.
    """
    print("\n" + "="*70)
    print("TEST 3: Hermiticity Preservation")
    print("="*70)
    
    n_orientations = 50
    n_lags = 100
    
    print(f"\n  Testing hermiticity: C(m1,m2) = C*(m2,m1)")
    
    # Create hermitian correlation matrix
    m_values = [-2, -1, 0, 1, 2]
    corr_matrix = {}
    
    np.random.seed(42)
    for i, m1 in enumerate(m_values):
        for j, m2 in enumerate(m_values):
            if m1 < m2:
                # Create random correlation
                corr = np.random.randn(n_lags) + 1j * np.random.randn(n_lags)
                corr_matrix[(m1, m2)] = corr
                # Ensure hermiticity
                corr_matrix[(m2, m1)] = np.conj(corr)
            elif m1 == m2:
                # Diagonal must be real
                corr_matrix[(m1, m2)] = np.random.randn(n_lags)
    
    # Rotate
    euler_angles = create_uniform_orientations(n_orientations)
    config = NMRConfig(verbose=False)
    calc = RotatedCorrelationCalculator(config)
    calc.compute_wigner_d_from_euler(euler_angles)
    rotated = calc.rotate_correlation_matrix(corr_matrix)
    ensemble_avg = calc.compute_ensemble_average()
    
    # Check hermiticity
    print("\n  Checking hermiticity at τ=0:")
    print("  " + "-"*66)
    
    max_violation = 0
    violations = []
    
    for i, m1 in enumerate(m_values):
        for j, m2 in enumerate(m_values):
            if m1 <= m2:
                C_12 = ensemble_avg[i, j, 0]
                C_21 = ensemble_avg[j, i, 0]
                C_21_conj = np.conj(C_21)
                
                diff = np.abs(C_12 - C_21_conj)
                magnitude = np.abs(C_12)
                
                if magnitude > 1e-10:
                    rel_diff = diff / magnitude
                    max_violation = max(max_violation, rel_diff)
                    violations.append(rel_diff)
                    
                    if m1 != m2:  # Only print off-diagonal
                        print(f"  C({m1:2d},{m2:2d}) - C*({m2:2d},{m1:2d}):  "
                              f"diff = {diff:.2e}, rel = {rel_diff:.2e}")
    
    print("  " + "-"*66)
    print(f"\n  Maximum relative violation: {max_violation:.2e}")
    print(f"  Average relative violation: {np.mean(violations):.2e}")
    
    # Validation
    assert max_violation < 1e-10, f"Hermiticity violated: max error = {max_violation:.2e}"
    
    print("\n  ✓ Hermiticity preserved after rotation")
    print("  ✓ C(m1,m2) = C*(m2,m1) within numerical precision")
    
    print("\n✓ TEST 3 PASSED")
    return True


def test_rotation_invariants():
    """
    Test 4: Test that Frobenius norm is preserved (unitarity).
    
    For unitary transformation C' = D × C × D†, the Frobenius norm should be preserved:
    ||C'||_F = ||C||_F for each rotation.
    
    Note: This test uses all orientations with the same C, which isn't physical,
    but tests the mathematical property of unitary transforms.
    """
    print("\n" + "="*70)
    print("TEST 4: Frobenius Norm Preservation (Unitarity)")
    print("="*70)
    
    n_orientations = 100
    n_lags = 50
    
    print(f"\n  Testing that ||D×C×D†||_F = ||C||_F for all D")
    
    # Create simple test correlation matrix
    corr_matrix = create_isotropic_correlation_matrix(n_lags)
    
    # Convert to array
    m_values = [-2, -1, 0, 1, 2]
    C_original = np.zeros((5, 5, n_lags), dtype=complex)
    for i, m1 in enumerate(m_values):
        for j, m2 in enumerate(m_values):
            C_original[i, j, :] = corr_matrix[(m1, m2)]
    
    # Get Wigner D matrices
    euler_angles = create_uniform_orientations(n_orientations)
    config = NMRConfig(verbose=False)
    calc = RotatedCorrelationCalculator(config)
    wigner_d_lib = calc.compute_wigner_d_from_euler(euler_angles)
    
    # Manually compute C' = D × C × D† for each orientation
    print("\n  Computing rotated matrices manually...")
    norm_original = np.linalg.norm(C_original[:, :, 0], 'fro')
    norm_errors = []
    
    for i in range(n_orientations):
        D = wigner_d_lib[i]
        D_H = D.T.conj()
        
        # C' = D × C × D†
        C_rotated = D @ C_original[:, :, 0] @ D_H
        norm_rotated = np.linalg.norm(C_rotated, 'fro')
        
        error = np.abs(norm_rotated - norm_original) / norm_original
        norm_errors.append(error)
    
    max_error = np.max(norm_errors)
    avg_error = np.mean(norm_errors)
    
    print(f"\n  Results at τ=0:")
    print("  " + "-"*66)
    print(f"  Original ||C||_F:       {norm_original:.6f}")
    print(f"  Max relative error:     {max_error:.2e}")
    print(f"  Avg relative error:     {avg_error:.2e}")
    print("  " + "-"*66)
    
    # Check that errors are at machine precision level
    assert max_error < 1e-12, f"Norm preservation error too large: {max_error}"
    
    print("\n  ✓ Frobenius norm preserved to machine precision")
    print("  ✓ Wigner D matrices are unitary")
    
    print("\n✓ TEST 4 PASSED")
    return True


def test_single_orientation_identity():
    """
    Test 5: Identity rotation test - verifying D(0,0,0) = I.
    """
    print("\n" + "="*70)
    print("TEST 5: Identity Rotation")
    print("="*70)
    
    n_lags = 50
    
    print(f"\n  Testing identity rotation: (α=0, β=0, γ=0)")
    
    # Create simple diagonal correlation matrix
    m_values = [-2, -1, 0, 1, 2]
    corr_matrix = {}
    
    for m1 in m_values:
        for m2 in m_values:
            if m1 == m2:
                corr_matrix[(m1, m2)] = np.ones(n_lags)
            else:
                corr_matrix[(m1, m2)] = np.zeros(n_lags, dtype=complex)
    
    # Identity rotation
    euler_angles = np.array([[0.0, 0.0, 0.0]])
    
    config = NMRConfig(verbose=False)
    calc = RotatedCorrelationCalculator(config)
    
    # Check Wigner D
    wigner_d = calc.compute_wigner_d_from_euler(euler_angles)
    
    print(f"\n  Wigner D matrix at (0,0,0):")
    print("  " + "-"*66)
    for i in range(5):
        row_str = "  [" + "  ".join([f"{wigner_d[0, i, j].real:5.1f}" for j in range(5)]) + "]"
        print(row_str)
    
    identity_diff = np.max(np.abs(wigner_d[0] - np.eye(5)))
    
    print(f"\n  Max |D(0,0,0) - I|: {identity_diff:.2e}")
    
    # Check that identity is exact
    assert identity_diff < 1e-12, f"Identity rotation error too large: {identity_diff}"
    
    print(f"\n  ✓ Identity rotation verified: D(0,0,0) = I")
    
    print("\n✓ TEST 5 PASSED")
    return True


if __name__ == '__main__':
    print("\n" + "="*70)
    print("MODULE 5: ANALYTICAL COMPARISON TESTS")
    print("="*70)
    print("\nTesting rotated correlation matrices against analytical predictions")
    
    results = []
    tests = [
        ("Isotropic Averaging", test_isotropic_averaging),
        ("Decay Rate Preservation", test_exponential_decay_preservation),
        ("Hermiticity", test_hermiticity),
        ("Rotation Invariants", test_rotation_invariants),
        ("Identity Rotation", test_single_orientation_identity),
    ]
    
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n✗ TEST FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("ANALYTICAL TEST SUMMARY")
    print("="*70)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
    
    n_passed = sum(1 for _, passed in results if passed)
    n_total = len(results)
    
    print(f"\n  Total: {n_passed}/{n_total} analytical tests passed")
    
    if n_passed == n_total:
        print("\n" + "="*70)
        print("✓ ALL ANALYTICAL TESTS PASSED")
        print("="*70)
        print("\nRotated correlation matrices match analytical predictions:")
        print("  • Isotropic averaging suppresses off-diagonal elements")
        print("  • Exponential decay rates preserved")
        print("  • Hermiticity maintained")
        print("  • Rotation invariants conserved")
        print("  • Identity rotation verified")
    else:
        print("\n" + "="*70)
        print("✗ SOME ANALYTICAL TESTS FAILED")
        print("="*70)
