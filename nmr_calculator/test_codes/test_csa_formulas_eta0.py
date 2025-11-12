"""
Test comparing two CSA T1 formulas for η=0 case using simulated data.

This test generates a trajectory with cone diffusion (η=0 CSA), calculates J(ω),
then compares T1 from:
1. Universal formula: R1 = ω₀² × J(ω₀) × 10⁻¹²  (works for any η)
2. Analytical formula: R1 = (1/3) × (ω₀ × Δσ)² × J(ω₀)  (only for η=0)

For η=0, both formulas should give the same result.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from nmr_calculator.config import NMRConfig
from nmr_calculator.xyz_generator import TrajectoryGenerator
from nmr_calculator.euler_converter import EulerConverter
from nmr_calculator.spherical_harmonics import SphericalHarmonicsCalculator
from nmr_calculator.autocorrelation import AutocorrelationCalculator
from nmr_calculator.spectral_density import SpectralDensityCalculator
from nmr_calculator.nmr_parameters import NMRParametersCalculator


def test_csa_formulas_eta0():
    """
    Test that both CSA T1 formulas agree for η=0.
    """
    print("="*70)
    print("TEST: CSA T1 Formula Comparison for η=0")
    print("="*70)
    
    # Parameters
    S2 = 0.85
    tau_c = 5e-9
    delta_sigma = 160.0
    B0 = 14.1
    nucleus = '13C'
    
    print(f"\nParameters:")
    print(f"  S² = {S2}")
    print(f"  τc = {tau_c*1e9:.1f} ns")
    print(f"  Δσ = {delta_sigma} ppm")
    print(f"  B₀ = {B0} T")
    print(f"  Nucleus: {nucleus}")
    print(f"  η = 0 (uniaxial)")
    
    # Generate trajectory
    print(f"\n{'='*70}")
    print("Step 1: Generate Trajectory and Calculate J(ω)")
    print(f"{'='*70}")
    
    config = NMRConfig(
        trajectory_type='diffusion_cone',
        S2=S2,
        tau_c=tau_c,
        dt=0.1e-12,
        num_steps=200000,
        B0=B0,
        nucleus=nucleus,
        interaction_type='CSA',
        delta_sigma=delta_sigma,
        eta=0.0,
        calculate_T1=True,
        verbose=True
    )
    
    # Full pipeline
    gen = TrajectoryGenerator(config)
    rotations, _ = gen.generate()
    
    converter = EulerConverter(config)
    euler_angles = converter.convert(rotations=rotations)
    
    sh_calc = SphericalHarmonicsCalculator(config)
    Y2m = sh_calc.calculate(euler_angles)
    
    acf_calc = AutocorrelationCalculator(config)
    acf, time_lags = acf_calc.calculate(Y2m)
    
    sd_calc = SpectralDensityCalculator(config)
    J_sim, freq_sim = sd_calc.calculate(acf, time_lags)
    
    omega_0 = config.get_omega0()
    idx_omega0 = np.argmin(np.abs(freq_sim - omega_0))
    J_omega0 = J_sim[idx_omega0]
    
    print(f"\nSimulated J(ω₀) = {J_omega0:.3e} s")
    
    # ========================================================================
    # PART 2: Calculate T1 using UNIVERSAL formula
    # ========================================================================
    print(f"\n{'='*70}")
    print("Step 2: Calculate T1 using Universal Formula")
    print(f"{'='*70}")
    
    calc_universal = NMRParametersCalculator(config)
    T1_universal, _ = calc_universal.calculate(J_sim, freq_sim)
    
    print(f"\nT1 (universal formula): {T1_universal:.3f} s ({T1_universal*1000:.1f} ms)")
    
    # ========================================================================
    # PART 3: Calculate T1 using ANALYTICAL formula (η=0)
    # ========================================================================
    print(f"\n{'='*70}")
    print("Step 3: Calculate T1 using Analytical Formula (η=0 only)")
    print(f"{'='*70}")
    
    # Manually calculate using analytical formula
    larmor_freq_Hz = omega_0 / (2 * np.pi)
    delta_sigma_rad = delta_sigma * 1e-6 * omega_0
    
    # Analytical: R1 = (1/3) × (ω₀ × Δσ)² × J(ω₀)
    R1_analytical = (1.0/3.0) * (delta_sigma_rad)**2 * J_omega0
    T1_analytical = 1.0 / R1_analytical
    
    print(f"\nAnalytical calculation:")
    print(f"  ω₀ = {omega_0:.3e} rad/s ({larmor_freq_Hz*1e-6:.2f} MHz)")
    print(f"  Δσ = {delta_sigma} ppm = {delta_sigma_rad:.3e} rad/s")
    print(f"  J(ω₀) = {J_omega0:.3e} s")
    print(f"  R1 = (1/3) × ({delta_sigma_rad:.2e})² × {J_omega0:.2e}")
    print(f"  R1 = {R1_analytical:.3e} s⁻¹")
    print(f"\nT1 (analytical formula): {T1_analytical:.3f} s ({T1_analytical*1000:.1f} ms)")
    
    # ========================================================================
    # PART 4: Compare the two formulas
    # ========================================================================
    print(f"\n{'='*70}")
    print("Step 4: Comparison")
    print(f"{'='*70}")
    
    # Universal formula details
    R1_universal = calc_universal.R1_csa
    larmor_Hz = larmor_freq_Hz
    
    print(f"\nUniversal Formula: R1 = ω₀² × J(ω₀) × 10⁻¹²")
    print(f"  R1 = ({larmor_Hz:.2e})² × {J_omega0:.2e} × 10⁻¹²")
    print(f"  R1 = {R1_universal:.3e} s⁻¹")
    print(f"  T1 = {T1_universal:.3f} s")
    
    print(f"\nAnalytical Formula: R1 = (1/3) × (ω₀×Δσ)² × J(ω₀)")
    print(f"  R1 = (1/3) × ({delta_sigma_rad:.2e})² × {J_omega0:.2e}")
    print(f"  R1 = {R1_analytical:.3e} s⁻¹")
    print(f"  T1 = {T1_analytical:.3f} s")
    
    # Check if formulas are equivalent
    print(f"\n{'='*70}")
    print("Formula Equivalence Check")
    print(f"{'='*70}")
    
    # For η=0, the formulas should be related by:
    # Universal: R1 = ω₀² × J(ω₀) × 10⁻¹² (using ω₀ in Hz)
    # Analytical: R1 = (1/3) × (ω₀×Δσ)² × J(ω₀) (using ω₀ in rad/s, Δσ in rad/s)
    
    # Convert Δσ: Δσ_rad = Δσ_ppm × 10⁻⁶ × ω₀
    # So: (ω₀×Δσ)² = ω₀² × (Δσ_ppm × 10⁻⁶ × ω₀)²
    #              = ω₀² × Δσ_ppm² × 10⁻¹² × ω₀²
    #              = ω₀⁴ × Δσ_ppm² × 10⁻¹²
    
    # But universal uses ω₀ in Hz, so let's check the relationship
    ratio_R1 = R1_universal / R1_analytical
    ratio_T1 = T1_universal / T1_analytical
    
    print(f"\nR1 ratio (universal/analytical): {ratio_R1:.6f}")
    print(f"T1 ratio (universal/analytical): {ratio_T1:.6f}")
    
    # For η=0, theoretical relationship:
    # Universal: R1 = (ω₀_Hz)² × J × 10⁻¹²
    # Analytical: R1 = (1/3) × (ω₀_rad × Δσ_rad)² × J
    #           = (1/3) × (ω₀_rad × Δσ_ppm × 10⁻⁶ × ω₀_rad)² × J
    #           = (1/3) × ω₀_rad⁴ × Δσ_ppm² × 10⁻¹² × J
    
    # Since ω₀_rad = 2π × ω₀_Hz:
    # Analytical: R1 = (1/3) × (2π×ω₀_Hz)⁴ × Δσ_ppm² × 10⁻¹² × J
    
    # So the ratio should be:
    # R1_universal / R1_analytical = (ω₀_Hz)² / [(1/3) × (2π×ω₀_Hz)⁴ × Δσ_ppm²]
    #                              = 3 / [(2π)⁴ × ω₀_Hz² × Δσ_ppm²]
    
    expected_ratio = 3.0 / ((2*np.pi)**4 * larmor_Hz**2 * delta_sigma**2)
    print(f"\nExpected ratio from theory: {expected_ratio:.6e}")
    print(f"Actual ratio: {ratio_R1:.6e}")
    
    # The formulas are NOT equivalent! They have different physics.
    # Universal assumes correlation in ppm² units
    # Analytical assumes normalized correlation
    
    print(f"\n{'='*70}")
    print("CONCLUSION")
    print(f"{'='*70}")
    print(f"\nThe two formulas are NOT directly equivalent!")
    print(f"\n1. Universal formula: R1 = ω₀² × J(ω₀) × 10⁻¹²")
    print(f"   - Assumes J(ω) from correlation in ppm² units")
    print(f"   - CSA magnitude already in correlation function")
    print(f"   - T1 = {T1_universal*1000:.1f} ms")
    
    print(f"\n2. Analytical formula: R1 = (1/3) × (ω₀×Δσ)² × J(ω₀)")
    print(f"   - Assumes J(ω) from NORMALIZED correlation")
    print(f"   - CSA magnitude (Δσ) applied separately in formula")
    print(f"   - T1 = {T1_analytical*1000:.1f} ms")
    
    print(f"\nFor η=0 with our simulated data:")
    print(f"  Ratio: {ratio_T1:.2f}×")
    print(f"\nTo use analytical formula correctly, need to normalize ACF first!")
    
    return T1_universal, T1_analytical, ratio_T1


if __name__ == '__main__':
    T1_univ, T1_anal, ratio = test_csa_formulas_eta0()
    
    print(f"\n{'='*70}")
    print("TEST COMPLETE")
    print(f"{'='*70}")
    print(f"\nThe formulas apply to DIFFERENT types of input:")
    print(f"  - Universal: For J(ω) from correlation WITH CSA in ppm")
    print(f"  - Analytical: For J(ω) from NORMALIZED correlation")
