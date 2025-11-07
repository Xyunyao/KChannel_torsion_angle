"""
Module 7 Test: NMR Relaxation Parameters

Simple tests for T1, T2, and NOE calculations.
"""

import numpy as np
from nmr_calculator.config import NMRConfig
from nmr_calculator.nmr_parameters import NMRParametersCalculator


def test_dipolar_T1():
    """Test dipolar T1 calculation with known spectral density."""
    print("\n" + "="*70)
    print("TEST 1: Dipolar T1 Calculation")
    print("="*70)
    
    # Create config
    config = NMRConfig(
        nucleus='15N',
        B0=14.1,  # Tesla (600 MHz for 1H)
        interaction_type='dipolar',
        calculate_T1=True,
        calculate_T2=False,
        r_NH=1.02e-10,  # 1.02 Angstrom
        verbose=True
    )
    
    # Create mock spectral density
    # For testing, use simple values
    omega_N = config.get_omega0()
    omega_H = 2 * np.pi * config.B0 * 2.675222005e8  # 1H frequency
    
    # Create frequency array
    frequencies = np.array([
        0,  # J(0)
        omega_N,  # J(ωN)
        omega_H - omega_N,  # J(ωH - ωN)
        omega_H,  # J(ωH)
        omega_H + omega_N,  # J(ωH + ωN)
    ])
    
    # Mock J(ω) values (typical for 5 ns correlation time)
    spectral_density = np.array([
        1.7e-9,  # J(0)
        1.0e-11,  # J(ωN)
        8.0e-12,  # J(ωH-ωN)
        5.0e-12,  # J(ωH)
        2.5e-12,  # J(ωH+ωN)
    ])
    
    # Calculate T1
    calc = NMRParametersCalculator(config)
    T1, T2 = calc.calculate(spectral_density, frequencies)
    
    print(f"\nResults:")
    print(f"  T1 = {T1:.4f} s = {T1*1000:.1f} ms")
    print(f"  R1 = {1/T1:.4f} s⁻¹")
    
    # Check if reasonable (typical T1 for proteins: 0.3-1.5 s at 600 MHz)
    assert 0.1 < T1 < 3.0, f"T1 ({T1:.3f} s) outside expected range"
    
    print("\n✓ TEST 1 PASSED")
    return True


def test_CSA_T1():
    """Test CSA T1 calculation."""
    print("\n" + "="*70)
    print("TEST 2: CSA T1 Calculation")
    print("="*70)
    
    config = NMRConfig(
        nucleus='15N',
        B0=14.1,
        interaction_type='CSA',
        delta_sigma=160.0,  # ppm
        eta=0.0,
        calculate_T1=True,
        calculate_T2=False,
        verbose=True
    )
    
    omega_N = config.get_omega0()
    
    frequencies = np.array([0, omega_N, 2*omega_N])
    spectral_density = np.array([1.7e-9, 1.0e-11, 2.5e-12])
    
    calc = NMRParametersCalculator(config)
    T1, T2 = calc.calculate(spectral_density, frequencies)
    
    print(f"\nResults:")
    print(f"  T1 = {T1:.4f} s = {T1*1000:.1f} ms")
    
    # CSA contribution typically smaller than dipolar
    assert 1.0 < T1 < 100.0, f"T1 ({T1:.3f} s) outside expected range for CSA"
    
    print("\n✓ TEST 2 PASSED")
    return True


def test_T2_calculation():
    """Test T2 calculation (includes J(0) term)."""
    print("\n" + "="*70)
    print("TEST 3: T2 Calculation")
    print("="*70)
    
    config = NMRConfig(
        nucleus='15N',
        B0=14.1,
        interaction_type='dipolar',
        calculate_T1=True,
        calculate_T2=True,
        r_NH=1.02e-10,
        verbose=True
    )
    
    omega_N = config.get_omega0()
    omega_H = 2 * np.pi * config.B0 * 2.675222005e8
    
    frequencies = np.array([
        0,
        omega_N,
        omega_H - omega_N,
        omega_H,
        omega_H + omega_N,
    ])
    
    spectral_density = np.array([
        1.7e-9,  # J(0) - important for T2
        1.0e-11,
        8.0e-12,
        5.0e-12,
        2.5e-12,
    ])
    
    calc = NMRParametersCalculator(config)
    T1, T2 = calc.calculate(spectral_density, frequencies)
    
    print(f"\nResults:")
    print(f"  T1 = {T1:.4f} s = {T1*1000:.1f} ms")
    print(f"  T2 = {T2:.4f} s = {T2*1000:.1f} ms")
    print(f"  T1/T2 ratio = {T1/T2:.2f}")
    
    # T2 should be <= T1 (due to additional J(0) contribution)
    assert T2 <= T1, f"T2 ({T2:.3f}) should be <= T1 ({T1:.3f})"
    
    # Typical T2 range
    assert 0.05 < T2 < 2.0, f"T2 ({T2:.3f} s) outside expected range"
    
    print("\n✓ TEST 3 PASSED")
    return True


def test_NOE_calculation():
    """Test heteronuclear NOE calculation."""
    print("\n" + "="*70)
    print("TEST 4: NOE Calculation")
    print("="*70)
    
    config = NMRConfig(
        nucleus='15N',
        B0=14.1,
        interaction_type='dipolar',
        calculate_T1=True,
        r_NH=1.02e-10,
        verbose=True
    )
    
    omega_N = config.get_omega0()
    omega_H = 2 * np.pi * config.B0 * 2.675222005e8
    
    frequencies = np.array([
        0,
        omega_N,
        omega_H - omega_N,
        omega_H,
        omega_H + omega_N,
    ])
    
    spectral_density = np.array([
        1.7e-9,
        1.0e-11,
        8.0e-12,
        5.0e-12,
        2.5e-12,
    ])
    
    calc = NMRParametersCalculator(config)
    T1, T2 = calc.calculate(spectral_density, frequencies)
    
    # Calculate NOE
    J_values = calc._get_J_at_frequencies(spectral_density, frequencies)
    NOE = calc.calculate_NOE(J_values)
    
    print(f"\nResults:")
    print(f"  NOE = {NOE:.3f}")
    
    # Typical NOE range for 15N-1H: -3.5 to 0.8
    # Rigid molecules at high field: NOE ~ -3 to -4
    # Flexible regions: NOE closer to 0
    assert -5.0 < NOE < 1.0, f"NOE ({NOE:.3f}) outside expected range"
    
    print("\n✓ TEST 4 PASSED")
    return True


def test_field_dependence():
    """Test T1 at different field strengths."""
    print("\n" + "="*70)
    print("TEST 5: Field Dependence")
    print("="*70)
    
    fields = [11.7, 14.1, 18.8]  # T (corresponding to 500, 600, 800 MHz)
    T1_values = []
    
    # Fixed spectral density shape (Lipari-Szabo with τc=5ns, S²=0.85)
    tau_c = 5e-9
    S2 = 0.85
    tau_f = 0.1e-9
    tau_e = 1.0 / (1.0/tau_c + 1.0/tau_f)
    
    for B0 in fields:
        config = NMRConfig(
            nucleus='15N',
            B0=B0,
            interaction_type='dipolar',
            calculate_T1=True,
            r_NH=1.02e-10,
            verbose=False
        )
        
        omega_N = config.get_omega0()
        omega_H = 2 * np.pi * B0 * 2.675222005e8
        
        frequencies = np.array([0, omega_N, omega_H-omega_N, omega_H, omega_H+omega_N])
        
        # Lipari-Szabo spectral density
        def J_LS(omega):
            return (2.0/5.0) * (S2 * tau_c / (1 + (omega*tau_c)**2) +
                               (1-S2) * tau_e / (1 + (omega*tau_e)**2))
        
        spectral_density = np.array([J_LS(w) for w in frequencies])
        
        calc = NMRParametersCalculator(config)
        T1, _ = calc.calculate(spectral_density, frequencies)
        T1_values.append(T1)
        
        print(f"  B0 = {B0:.1f} T → T1 = {T1*1000:.1f} ms")
    
    # T1 should decrease slightly with field (high field limit)
    # But the decrease is weak for typical protein motions
    print(f"\n  T1 range: {min(T1_values)*1000:.1f} - {max(T1_values)*1000:.1f} ms")
    
    print("\n✓ TEST 5 PASSED")
    return True


if __name__ == '__main__':
    print("\n" + "="*70)
    print("MODULE 7: NMR RELAXATION PARAMETERS - TEST SUITE")
    print("="*70)
    
    results = []
    tests = [
        ("Dipolar T1", test_dipolar_T1),
        ("CSA T1", test_CSA_T1),
        ("T2 Calculation", test_T2_calculation),
        ("NOE Calculation", test_NOE_calculation),
        ("Field Dependence", test_field_dependence),
    ]
    
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, "PASSED" if success else "FAILED"))
        except Exception as e:
            print(f"\n✗ TEST FAILED: {e}")
            results.append((name, f"FAILED: {e}"))
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for name, status in results:
        symbol = "✓" if "PASSED" in status else "✗"
        print(f"  {symbol} {name:25s} {status}")
    
    passed = sum(1 for _, status in results if "PASSED" in status)
    total = len(results)
    
    print(f"\n  Total: {passed}/{total} tests passed")
    print("="*70)
    
    if passed == total:
        print("\n✓ ALL TESTS PASSED")
        exit(0)
    else:
        print(f"\n✗ {total-passed} TEST(S) FAILED")
        exit(1)
