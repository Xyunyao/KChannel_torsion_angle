"""
Test comparing analytical CSA T1 (uniaxial, η=0) vs universal CSA T1 formula.

This test uses the dynamics model: global tumbling + local diffusion on a cone
For η=0 CSA tensor, both methods should give similar T1 values.

Workflow:
1. Simulate trajectory with cone diffusion (local) + isotropic tumbling (global)
2. Calculate Y2m with CSA tensor (η=0)
3. Compute autocorrelation function
4. Calculate spectral density
5. Compare T1 from:
   - Universal formula: R1 = ω₀² × J(ω₀) × 10⁻¹²
   - Analytical formula: R1 = (1/3) × (ω₀ × Δσ)² × J(ω₀)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import matplotlib.pyplot as plt
from nmr_calculator.config import NMRConfig
from nmr_calculator.xyz_generator import TrajectoryGenerator
from nmr_calculator.euler_converter import EulerConverter
from nmr_calculator.spherical_harmonics import SphericalHarmonicsCalculator
from nmr_calculator.autocorrelation import AutocorrelationCalculator
from nmr_calculator.spectral_density import SpectralDensityCalculator
from nmr_calculator.nmr_parameters import NMRParametersCalculator


def lipari_szabo_spectral_density(omega, S2, tau_c, tau_f):
    """
    Analytical Lipari-Szabo spectral density.
    
    J(ω) = (2/5) × [S² × τc/(1 + ω²τc²) + (1-S²) × τ/(1 + ω²τ²)]
    
    where τ = τc × τf / (τc + τf)
    
    Parameters
    ----------
    omega : float or ndarray
        Angular frequency (rad/s)
    S2 : float
        Order parameter (0 to 1)
    tau_c : float
        Global correlation time (seconds)
    tau_f : float
        Fast internal motion correlation time (seconds)
    
    Returns
    -------
    J : float or ndarray
        Spectral density
    """
    # Effective correlation time for fast motion
    tau_eff = (tau_c * tau_f) / (tau_c + tau_f)
    
    # Lipari-Szabo formula
    J = (2.0/5.0) * (
        S2 * tau_c / (1 + (omega * tau_c)**2) + 
        (1 - S2) * tau_eff / (1 + (omega * tau_eff)**2)
    )
    
    return J


def test_csa_t1_eta0_comparison():
    """
    Test CSA T1 calculation comparing analytical vs universal formula.
    Uses cone diffusion + global tumbling with η=0.
    
    NOTE: This test compares the two formulas using the SAME simulated J(ω),
    not comparing analytical Lipari-Szabo vs simulated trajectory.
    """
    print("="*70)
    print("TEST: CSA T1 Formula Comparison (Analytical vs Universal)")
    print("Using identical simulated J(ω) with η=0")
    print("="*70)
    
    # Physical parameters
    S2 = 0.85           # Order parameter
    tau_c = 5e-9        # Global correlation time (5 ns)
    tau_f = 0.1e-9      # Fast motion (100 ps) - NOTE: used for analytical only
    delta_sigma = 160.0 # CSA anisotropy (ppm)
    B0 = 14.1           # Magnetic field (T) - 600 MHz for 1H
    nucleus = '13C'     # Carbon-13
    
    print(f"\nPhysical Parameters:")
    print(f"  S² = {S2}")
    print(f"  τc = {tau_c*1e9:.1f} ns (global tumbling)")
    print(f"  τf = {tau_f*1e9:.1f} ps (local motion - analytical only)")
    print(f"  NOTE: Cone diffusion model uses S² and τc, not τf explicitly")
    print(f"  Δσ = {delta_sigma} ppm")
    print(f"  B₀ = {B0} T")
    print(f"  Nucleus: {nucleus}")
    print(f"  η = 0 (uniaxial CSA)")
    
    # Simulation parameters
    dt = 0.1e-12        # 0.1 ps time step
    num_steps = 200000  # 20 ns trajectory
    
    print(f"\nSimulation Parameters:")
    print(f"  dt = {dt*1e12:.1f} ps")
    print(f"  Total steps = {num_steps:,}")
    print(f"  Total time = {num_steps*dt*1e9:.1f} ns")
    
    # ========================================================================
    # PART 1: Generate simulated trajectory and calculate J(ω)
    # ========================================================================
    print(f"\n{'='*70}")
    print("PART 1: Generate Trajectory and Calculate J(ω)")
    print(f"{'='*70}")
    
    # Create config for analytical calculation
    config_analytical = NMRConfig(
        B0=B0,
        nucleus=nucleus,
        interaction_type='CSA',
        delta_sigma=delta_sigma,
        eta=0.0,
        calculate_T1=True,
        calculate_T2=False,
        verbose=True
    )
    
    omega_0 = config_analytical.get_omega0()
    larmor_freq = omega_0 / (2 * np.pi)
    
    print(f"\nLarmor frequency: {larmor_freq*1e-6:.2f} MHz")
    
    # Calculate analytical J(ω) at ω₀
    J_omega0_analytical = lipari_szabo_spectral_density(omega_0, S2, tau_c, tau_f)
    
    print(f"\nAnalytical Lipari-Szabo:")
    print(f"  J(ω₀) = {J_omega0_analytical:.3e} s")
    
    # Create mock spectral density for analytical method
    frequencies_analytical = np.array([0, omega_0])
    spectral_density_analytical = np.array([
        lipari_szabo_spectral_density(0, S2, tau_c, tau_f),
        J_omega0_analytical
    ])
    
    # Calculate T1 using analytical uniaxial formula
    calc_analytical = NMRParametersCalculator(config_analytical)
    
    # Manually call the uniaxial method
    J_values = {'J_omega_N': J_omega0_analytical, 'J_0': spectral_density_analytical[0]}
    T1_analytical = calc_analytical._calculate_T1_CSA_uniaxial(J_values)
    
    print(f"\nT1 (analytical uniaxial formula):")
    print(f"  T1 = {T1_analytical:.3f} s ({T1_analytical*1000:.1f} ms)")
    
    # ========================================================================
    # PART 2: Simulated T1 from trajectory using universal formula
    # ========================================================================
    print(f"\n{'='*70}")
    print("PART 2: Simulated T1 (Trajectory + Universal Formula)")
    print(f"{'='*70}")
    
    # Create config for simulation
    config_sim = NMRConfig(
        trajectory_type='diffusion_cone',
        S2=S2,
        tau_c=tau_c,
        dt=dt,
        num_steps=num_steps,
        B0=B0,
        nucleus=nucleus,
        interaction_type='CSA',
        delta_sigma=delta_sigma,
        eta=0.0,  # Uniaxial
        calculate_T1=True,
        calculate_T2=False,
        verbose=True
    )
    
    # Generate trajectory
    print("\nStep 1: Generate trajectory (cone diffusion + global tumbling)...")
    gen = TrajectoryGenerator(config_sim)
    rotations, _ = gen.generate()
    
    # Convert to Euler angles
    print("\nStep 2: Convert to Euler angles...")
    converter = EulerConverter(config_sim)
    euler_angles = converter.convert(rotations=rotations)
    
    # Calculate Y2m with CSA
    print("\nStep 3: Calculate Y2m with CSA tensor...")
    sh_calc = SphericalHarmonicsCalculator(config_sim)
    Y2m = sh_calc.calculate(euler_angles)
    
    # Calculate autocorrelation
    print("\nStep 4: Calculate autocorrelation function...")
    acf_calc = AutocorrelationCalculator(config_sim)
    acf, time_lags = acf_calc.calculate(Y2m)
    
    # Calculate spectral density
    print("\nStep 5: Calculate spectral density...")
    sd_calc = SpectralDensityCalculator(config_sim)
    J_sim, freq_sim = sd_calc.calculate(acf, time_lags)
    
    # Calculate T1 using universal formula
    print("\nStep 6: Calculate T1 using universal formula...")
    calc_sim = NMRParametersCalculator(config_sim)
    T1_sim, _ = calc_sim.calculate(J_sim, freq_sim)
    
    print(f"\nT1 (universal formula from simulation):")
    print(f"  T1 = {T1_sim:.3f} s ({T1_sim*1000:.1f} ms)")
    
    # ========================================================================
    # PART 3: Comparison
    # ========================================================================
    print(f"\n{'='*70}")
    print("PART 3: Comparison")
    print(f"{'='*70}")
    
    # Get J(ω₀) from simulation
    idx_omega0 = np.argmin(np.abs(freq_sim - omega_0))
    J_omega0_sim = J_sim[idx_omega0]
    
    print(f"\nSpectral Density at ω₀:")
    print(f"  Analytical J(ω₀):  {J_omega0_analytical:.3e} s")
    print(f"  Simulated J(ω₀):   {J_omega0_sim:.3e} s")
    print(f"  Ratio (sim/ana):   {J_omega0_sim/J_omega0_analytical:.3f}")
    
    print(f"\nT1 Values:")
    print(f"  Analytical (uniaxial):  {T1_analytical:.3f} s ({T1_analytical*1000:.1f} ms)")
    print(f"  Simulated (universal):  {T1_sim:.3f} s ({T1_sim*1000:.1f} ms)")
    
    # Calculate relative difference
    rel_diff = abs(T1_sim - T1_analytical) / T1_analytical * 100
    print(f"  Relative difference:    {rel_diff:.1f}%")
    
    # Check if they agree within 20%
    if rel_diff < 20:
        print(f"\n  ✓ PASS: T1 values agree within 20%")
        test_passed = True
    else:
        print(f"\n  ✗ FAIL: T1 values differ by more than 20%")
        test_passed = False
    
    # ========================================================================
    # PART 4: Visualization
    # ========================================================================
    print(f"\n{'='*70}")
    print("PART 4: Visualization")
    print(f"{'='*70}")
    
    # Plot spectral densities
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Autocorrelation function
    ax = axes[0, 0]
    time_ns = time_lags * 1e9
    ax.plot(time_ns, acf.real, 'b-', alpha=0.7, label='Simulated ACF')
    
    # Analytical ACF (Lipari-Szabo)
    tau_eff = (tau_c * tau_f) / (tau_c + tau_f)
    acf_analytical = S2 * np.exp(-time_lags/tau_c) + (1-S2) * np.exp(-time_lags/tau_eff)
    ax.plot(time_ns, acf_analytical, 'r--', linewidth=2, label='Analytical (Lipari-Szabo)')
    
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('C(τ)')
    ax.set_title('Autocorrelation Function')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 5])  # First 5 ns
    
    # Plot 2: Spectral density (full range)
    ax = axes[0, 1]
    freq_MHz = freq_sim / (2 * np.pi * 1e6)
    ax.loglog(freq_MHz, J_sim, 'b-', alpha=0.7, label='Simulated J(ω)')
    
    # Analytical J(ω)
    freq_analytical = np.logspace(6, 11, 1000)  # 1 MHz to 100 GHz
    J_analytical = lipari_szabo_spectral_density(freq_analytical, S2, tau_c, tau_f)
    freq_analytical_MHz = freq_analytical / (2 * np.pi * 1e6)
    ax.loglog(freq_analytical_MHz, J_analytical, 'r--', linewidth=2, label='Analytical (Lipari-Szabo)')
    
    # Mark ω₀
    larmor_MHz = larmor_freq * 1e-6
    ax.axvline(larmor_MHz, color='k', linestyle=':', alpha=0.5, label=f'ω₀ = {larmor_MHz:.1f} MHz')
    ax.plot(larmor_MHz, J_omega0_analytical, 'ro', markersize=10, label=f'J(ω₀) analytical')
    ax.plot(larmor_MHz, J_omega0_sim, 'bs', markersize=8, label=f'J(ω₀) simulated')
    
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('J(ω) (s)')
    ax.set_title('Spectral Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Spectral density zoom near ω₀
    ax = axes[1, 0]
    # Find indices near ω₀
    freq_range = (freq_MHz > larmor_MHz * 0.5) & (freq_MHz < larmor_MHz * 2.0)
    ax.plot(freq_MHz[freq_range], J_sim[freq_range], 'b-', linewidth=2, label='Simulated')
    
    # Analytical near ω₀
    freq_zoom = np.linspace(omega_0*0.5, omega_0*2.0, 1000)
    J_zoom = lipari_szabo_spectral_density(freq_zoom, S2, tau_c, tau_f)
    ax.plot(freq_zoom/(2*np.pi*1e6), J_zoom, 'r--', linewidth=2, label='Analytical')
    
    ax.axvline(larmor_MHz, color='k', linestyle=':', alpha=0.5)
    ax.plot(larmor_MHz, J_omega0_analytical, 'ro', markersize=10)
    ax.plot(larmor_MHz, J_omega0_sim, 'bs', markersize=8)
    
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('J(ω) (s)')
    ax.set_title(f'Spectral Density near ω₀ ({larmor_MHz:.1f} MHz)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: T1 comparison bar chart
    ax = axes[1, 1]
    methods = ['Analytical\n(Uniaxial)', 'Simulated\n(Universal)']
    t1_values = [T1_analytical * 1000, T1_sim * 1000]  # in ms
    colors = ['red', 'blue']
    
    bars = ax.bar(methods, t1_values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('T1 (ms)')
    ax.set_title(f'T1 Comparison (η=0)\nDifference: {rel_diff:.1f}%')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, t1_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f} ms',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('nmr_calculator/test_csa_t1_comparison.png', dpi=300, bbox_inches='tight')
    print("\n  Saved plot: nmr_calculator/test_csa_t1_comparison.png")
    plt.show()
    
    # ========================================================================
    # Summary
    # ========================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\nFor η=0 CSA with S²={S2}, τc={tau_c*1e9:.1f}ns, τf={tau_f*1e9:.1f}ps:")
    print(f"  • Analytical formula (uniaxial):  T1 = {T1_analytical*1000:.1f} ms")
    print(f"  • Universal formula (simulated):  T1 = {T1_sim*1000:.1f} ms")
    print(f"  • Agreement: {rel_diff:.1f}% difference")
    print(f"\nBoth methods {'AGREE' if test_passed else 'DISAGREE'} for η=0 case!")
    
    return test_passed, T1_analytical, T1_sim, rel_diff


if __name__ == '__main__':
    test_passed, T1_ana, T1_sim, diff = test_csa_t1_eta0_comparison()
    
    if test_passed:
        print("\n" + "="*70)
        print("✓ TEST PASSED")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("✗ TEST FAILED")
        print("="*70)
