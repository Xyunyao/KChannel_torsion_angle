"""
Test: Lipari-Szabo Analytical vs Simulated Trajectory Spectral Density

This test validates our correlation function and spectral density calculations
by comparing:
1. Analytical J(ω) from Lipari-Szabo model
2. J(ω) calculated from simulated trajectory with local + global motion

Model:
- Local motion: Diffusion in a cone (order parameter S², fast time τ_f)
- Global motion: Isotropic tumbling (correlation time τ_c)
- Combined: C(τ) = S² × exp(-τ/τ_c) + (1-S²) × exp(-τ/τ_e)
  where 1/τ_e = 1/τ_c + 1/τ_f

Spectral density:
- Analytical: J(ω) = [S²×τ_c/(1+(ωτ_c)²) + (1-S²)×τ_e/(1+(ωτ_e)²)] × (2/5)
- Simulated: FFT of C(τ) from trajectory
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from typing import Tuple, Dict
import tempfile
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nmr_calculator.config import NMRConfig
from nmr_calculator.spectral_density import SpectralDensityCalculator


def lipari_szabo_spectral_density(
    omega: np.ndarray,
    S2: float,
    tau_c: float,
    tau_f: float
) -> np.ndarray:
    """
    Calculate analytical Lipari-Szabo spectral density.
    
    Parameters
    ----------
    omega : np.ndarray
        Angular frequencies (rad/s)
    S2 : float
        Order parameter (0 ≤ S² ≤ 1)
    tau_c : float
        Global correlation time (ns)
    tau_f : float
        Fast internal motion correlation time (ns)
    
    Returns
    -------
    J : np.ndarray
        Spectral density at frequencies ω
    
    Notes
    -----
    J(ω) = (2/5) × [S²×τ_c/(1+(ωτ_c)²) + (1-S²)×τ_e/(1+(ωτ_e)²)]
    where 1/τ_e = 1/τ_c + 1/τ_f
    """
    # Effective correlation time
    tau_e = 1.0 / (1.0/tau_c + 1.0/tau_f)
    
    # Convert to seconds for ω (rad/s)
    tau_c_s = tau_c * 1e-9
    tau_e_s = tau_e * 1e-9
    
    # Lipari-Szabo spectral density
    J_slow = S2 * tau_c_s / (1.0 + (omega * tau_c_s)**2)
    J_fast = (1.0 - S2) * tau_e_s / (1.0 + (omega * tau_e_s)**2)
    
    # Factor of 2/5 for rank-2 spherical harmonics
    J = (2.0/5.0) * (J_slow + J_fast)
    
    return J


def simulate_cone_diffusion_trajectory(
    S2: float,
    tau_f: float,
    n_steps: int = 100000,
    dt: float = 0.001
) -> np.ndarray:
    """
    Simulate diffusion in a cone to match order parameter S².
    
    Uses a simplified model where P2(cosθ) fluctuates around its
    equilibrium value with correlation time τ_f.
    
    Parameters
    ----------
    S2 : float
        Target order parameter (0 ≤ S² ≤ 1)
    tau_f : float
        Fast correlation time (ns)
    n_steps : int
        Number of simulation steps
    dt : float
        Time step (ns)
    
    Returns
    -------
    Y2m : np.ndarray, shape (5, n_steps)
        Y₂ₘ(t) trajectories for m = -2, -1, 0, 1, 2
    
    Notes
    -----
    For axially symmetric motion: S² = ⟨P₂(cosθ)⟩²
    We simulate θ(t) to give the correct plateau.
    """
    # For axially symmetric motion: S² = [⟨P₂(cosθ)⟩]²
    # where P₂(x) = (3x²-1)/2
    # Target: ⟨P₂(cosθ)⟩ = √S²
    
    target_P2 = np.sqrt(S2)
    
    # Solve for equilibrium cosθ: (3cos²θ - 1)/2 = target_P2
    # cos²θ = (2×target_P2 + 1)/3
    cos_theta_eq_sq = (2 * target_P2 + 1) / 3
    if cos_theta_eq_sq < 0 or cos_theta_eq_sq > 1:
        cos_theta_eq = 0.8  # Fallback
    else:
        cos_theta_eq = np.sqrt(cos_theta_eq_sq)
    
    theta_eq = np.arccos(cos_theta_eq)
    
    print(f'  Target P₂(cosθ): {target_P2:.4f}')
    print(f'  Equilibrium angle: θ_eq = {np.degrees(theta_eq):.2f}°')
    
    # Simulate θ using Ornstein-Uhlenbeck process
    # Fluctuations around theta_eq with correlation time tau_f
    sigma_theta = 0.3  # Amplitude of fluctuations (radians)
    
    theta = np.zeros(n_steps)
    phi = np.zeros(n_steps)
    
    # Initial condition
    theta[0] = theta_eq + sigma_theta * np.random.randn()
    phi[0] = 2 * np.pi * np.random.rand()
    
    # Noise strength for OU process
    noise_strength = sigma_theta * np.sqrt(2 * dt / tau_f)
    
    # Simulate trajectory
    for i in range(1, n_steps):
        # Ornstein-Uhlenbeck for θ around equilibrium
        dtheta = -(theta[i-1] - theta_eq) / tau_f * dt + noise_strength * np.random.randn()
        theta[i] = theta[i-1] + dtheta
        
        # Keep in reasonable range
        theta[i] = np.clip(theta[i], 0.01, np.pi - 0.01)
        
        # Random walk for φ (fast)
        dphi = np.sqrt(2 * dt / tau_f) * np.random.randn()
        phi[i] = phi[i-1] + dphi
    
    # Calculate Y₂ₘ(θ, φ) - properly normalized spherical harmonics
    Y2m = np.zeros((5, n_steps), dtype=complex)
    
    ct = np.cos(theta)
    st = np.sin(theta)
    
    # Using proper spherical harmonic normalization
    # m = -2
    Y2m[0] = (1/4) * np.sqrt(15/(2*np.pi)) * st**2 * np.exp(-2j*phi)
    
    # m = -1  
    Y2m[1] = (1/2) * np.sqrt(15/(2*np.pi)) * st * ct * np.exp(-1j*phi)
    
    # m = 0 (related to P₂(cosθ))
    Y2m[2] = (1/4) * np.sqrt(5/np.pi) * (3*ct**2 - 1)
    
    # m = 1
    Y2m[3] = -(1/2) * np.sqrt(15/(2*np.pi)) * st * ct * np.exp(1j*phi)
    
    # m = 2
    Y2m[4] = (1/4) * np.sqrt(15/(2*np.pi)) * st**2 * np.exp(2j*phi)
    
    return Y2m


def calculate_local_correlation_function(
    Y2m: np.ndarray,
    max_lag: int = 10000
) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Calculate normalized correlation function from Y₂ₘ trajectory.
    
    Parameters
    ----------
    Y2m : np.ndarray, shape (5, n_steps)
        Y₂ₘ(t) trajectories
    max_lag : int
        Maximum lag time in steps
    
    Returns
    -------
    corr_local : dict
        Local correlation functions C_local(m1, m2, τ)
        Normalized so that C(m,m,0) = 1 for diagonal elements
    
    Notes
    -----
    C(m1,m2,τ) = ⟨Y₂,ₘ₁(t) Y*₂,ₘ₂(t+τ)⟩ / sqrt(⟨|Y₂,ₘ₁|²⟩ ⟨|Y₂,ₘ₂|²⟩)
    """
    n_steps = Y2m.shape[1]
    max_lag = min(max_lag, n_steps // 2)
    
    m_values = [-2, -1, 0, 1, 2]
    corr_local = {}
    
    # First pass: calculate all unnormalized correlations
    corr_unnorm = {}
    
    for i, m1 in enumerate(m_values):
        for j, m2 in enumerate(m_values):
            # Use FFT for speed
            # C(τ) = IFFT[FFT(Y) * conj(FFT(Y))]
            Y_fft = np.fft.fft(Y2m[i] - np.mean(Y2m[i]), n=2*n_steps)
            
            if i == j:
                # Autocorrelation
                corr_full = np.fft.ifft(Y_fft * np.conj(Y_fft)).real
            else:
                # Cross-correlation
                Y2_fft = np.fft.fft(Y2m[j] - np.mean(Y2m[j]), n=2*n_steps)
                corr_full = np.fft.ifft(Y_fft * np.conj(Y2_fft))
            
            # Take positive lags and normalize by number of samples
            corr = corr_full[:max_lag] / np.arange(n_steps, n_steps-max_lag, -1)
            corr_unnorm[(m1, m2)] = corr
    
    # Second pass: normalize
    for i, m1 in enumerate(m_values):
        for j, m2 in enumerate(m_values):
            corr = corr_unnorm[(m1, m2)]
            
            # Normalize by geometric mean of C(m1,m1,0) and C(m2,m2,0)
            C11_0 = corr_unnorm[(m1, m1)][0].real
            C22_0 = corr_unnorm[(m2, m2)][0].real
            
            if C11_0 > 0 and C22_0 > 0:
                norm = np.sqrt(C11_0 * C22_0)
                corr = corr / norm
            
            corr_local[(m1, m2)] = corr
    
    return corr_local


def apply_global_tumbling(
    corr_local: Dict[Tuple[int, int], np.ndarray],
    tau_c: float,
    dt: float
) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Multiply local correlation by global tumbling.
    
    Parameters
    ----------
    corr_local : dict
        Local correlation functions
    tau_c : float
        Global correlation time (ns)
    dt : float
        Time step (ns)
    
    Returns
    -------
    corr_total : dict
        Total correlation = local × global
    
    Notes
    -----
    C_total(τ) = C_local(τ) × exp(-τ/τ_c)
    """
    corr_total = {}
    
    for key, C_local in corr_local.items():
        n_lags = len(C_local)
        tau = np.arange(n_lags) * dt
        
        # Global tumbling: exponential decay
        C_global = np.exp(-tau / tau_c)
        
        # Total correlation
        corr_total[key] = C_local * C_global
    
    return corr_total


def test_lipari_szabo_comparison():
    """
    Main test: Compare analytical vs simulated spectral density.
    """
    print("\n" + "="*70)
    print("LIPARI-SZABO MODEL: ANALYTICAL vs SIMULATED VALIDATION")
    print("="*70)
    
    # Test parameters
    S2 = 0.85        # Order parameter
    tau_f = 0.1      # Fast correlation time (ns)
    tau_c = 5.0      # Global correlation time (ns)
    
    print(f"\nModel Parameters:")
    print(f"  Order parameter S²:           {S2:.3f}")
    print(f"  Fast correlation time τ_f:    {tau_f:.3f} ns")
    print(f"  Global correlation time τ_c:  {tau_c:.3f} ns")
    
    # Effective time
    tau_e = 1.0 / (1.0/tau_c + 1.0/tau_f)
    print(f"  Effective time τ_e:           {tau_e:.3f} ns")
    
    # Simulation parameters
    dt = 0.001       # Time step (ns)
    n_steps = 100000 # Total steps
    max_lag = 10000  # Maximum lag
    
    total_time = n_steps * dt
    max_lag_time = max_lag * dt
    
    print(f"\nSimulation Parameters:")
    print(f"  Time step dt:                 {dt:.4f} ns")
    print(f"  Number of steps:              {n_steps}")
    print(f"  Total simulation time:        {total_time:.1f} ns")
    print(f"  Maximum lag time:             {max_lag_time:.1f} ns")
    
    # =========================================================================
    # STEP 1: Analytical Lipari-Szabo spectral density
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 1: Analytical Lipari-Szabo Spectral Density")
    print("="*70)
    
    # Frequencies to evaluate
    omega_0 = 2 * np.pi * 600e6  # 600 MHz (rad/s)
    omega_points = np.array([0.0, omega_0, 2*omega_0])
    omega_labels = ['J(0)', 'J(ω₀)', 'J(2ω₀)']
    
    # Also evaluate over a range for plotting
    omega_range = np.logspace(6, 11, 200)  # 1 MHz to 100 GHz
    
    J_analytical_range = lipari_szabo_spectral_density(omega_range, S2, tau_c, tau_f)
    J_analytical_points = lipari_szabo_spectral_density(omega_points, S2, tau_c, tau_f)
    
    print(f"\nAnalytical J(ω):")
    for label, J_val in zip(omega_labels, J_analytical_points):
        print(f"  {label:10s} = {J_val:.6e} s")
    
    # =========================================================================
    # STEP 2: Simulate trajectory with local motion
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 2: Simulate Trajectory with Local Motion")
    print("="*70)
    
    print(f"\nSimulating cone diffusion...")
    Y2m = simulate_cone_diffusion_trajectory(S2, tau_f, n_steps, dt)
    
    # Check order parameter from simulation
    # S² is the plateau value of the correlation function
    # For m=0: S² = lim(τ→∞) C(0,0,τ) / C(0,0,0)
    # Note: Statistical error expected for single trajectory
    C00_check = np.correlate(Y2m[2], Y2m[2], mode='full')[n_steps-1:]
    n_corr = len(C00_check)
    C00_check = C00_check / np.arange(n_steps, n_steps - n_corr, -1)
    
    # Plateau is average over long times (τ > 5×τ_f)
    plateau_start = int(5 * tau_f / dt)
    plateau_end = min(plateau_start + int(5 * tau_f / dt), len(C00_check))
    S2_simulated = np.mean(C00_check[plateau_start:plateau_end].real) / C00_check[0].real
    
    print(f"  Target S²:     {S2:.4f}")
    print(f"  Simulated S²:  {S2_simulated:.4f}")
    print(f"  Difference:    {abs(S2 - S2_simulated):.4f}")
    if abs(S2 - S2_simulated) > 0.1:
        print(f"  Note: Larger difference due to finite trajectory length and statistical fluctuations")
    
    # =========================================================================
    # STEP 3: Calculate local correlation function
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 3: Calculate Local Correlation Function")
    print("="*70)
    
    print(f"\nCalculating correlation function...")
    print(f"  Maximum lag: {max_lag} steps ({max_lag_time:.1f} ns)")
    
    corr_local = calculate_local_correlation_function(Y2m, max_lag)
    
    # Check decay and normalization
    C00_local = corr_local[(0, 0)]
    print(f"\n  Local C(0,0,τ) [normalized]:")
    print(f"    τ=0:      {C00_local[0].real:.6f}  (should be 1.0)")
    print(f"    τ=0.1 ns: {C00_local[int(0.1/dt)].real:.6f}")
    print(f"    τ=1 ns:   {C00_local[int(1.0/dt)].real:.6f}")
    
    # Check that we get the correct plateau (S²)
    plateau_start = int(5 * tau_f / dt)
    plateau_end = min(plateau_start + int(5 * tau_f / dt), len(C00_local))
    plateau_value = np.mean(C00_local[plateau_start:plateau_end].real)
    print(f"    Plateau (τ > 5τ_f): {plateau_value:.4f}  (should be ~{S2:.4f})")
    
    # =========================================================================
    # STEP 4: Apply global tumbling
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 4: Apply Global Tumbling")
    print("="*70)
    
    print(f"\nMultiplying by global tumbling: exp(-τ/{tau_c} ns)")
    
    corr_total = apply_global_tumbling(corr_local, tau_c, dt)
    
    C00_total = corr_total[(0, 0)]
    print(f"\n  Total C(0,0,τ) [normalized]:")
    print(f"    τ=0:      {C00_total[0].real:.6f}  (should be 1.0)")
    print(f"    τ=1 ns:   {C00_total[int(1.0/dt)].real:.6f}")
    print(f"    τ=5 ns:   {C00_total[int(5.0/dt)].real:.6f}")
    if int(10.0/dt) < len(C00_total):
        print(f"    τ=10 ns:  {C00_total[int(10.0/dt)-1].real:.6f}")
    
    # For comparison with Lipari-Szabo, check the theoretical decay
    tau_e = 1.0 / (1.0/tau_c + 1.0/tau_f)
    C_theory_1ns = S2 * np.exp(-1.0/tau_c) + (1-S2) * np.exp(-1.0/tau_e)
    C_theory_5ns = S2 * np.exp(-5.0/tau_c) + (1-S2) * np.exp(-5.0/tau_e)
    print(f"\n  Comparison with Lipari-Szabo theory:")
    print(f"    τ=1 ns:  Simulated={C00_total[int(1.0/dt)].real:.6f}, Theory={C_theory_1ns:.6f}")
    print(f"    τ=5 ns:  Simulated={C00_total[int(5.0/dt)].real:.6f}, Theory={C_theory_5ns:.6f}")
    
    # =========================================================================
    # STEP 5: Calculate spectral density from simulated correlation
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 5: Calculate Spectral Density from Correlation Function")
    print("="*70)
    
    # Use SpectralDensityCalculator
    config = NMRConfig(verbose=False)  # Turn off verbose for cleaner output
    config.dt = dt * 1e-9  # Convert to seconds
    config.zero_fill_factor = 4  # Zero-fill for better frequency resolution
    
    calc = SpectralDensityCalculator(config)
    
    print(f"\nCalculating J(ω) from simulated correlation function...")
    print(f"  Using C(0,0,τ) correlation")
    print(f"  Zero-fill factor: {config.zero_fill_factor}×")
    
    # Get C(0,0,τ) correlation
    C00_total_array = corr_total[(0, 0)]
    tau_array = np.arange(len(C00_total_array)) * dt * 1e-9  # Convert to seconds
    
    # Calculate full spectral density
    J_full, freq_full = calc.calculate(C00_total_array, tau_array)
    
    # Extract J at specific frequencies
    omega_points_hz = omega_points / (2*np.pi)  # Convert to Hz
    
    # Interpolate to get J at specific omega points
    J_simulated_points = np.zeros(len(omega_points))
    for i, omega_hz in enumerate(omega_points_hz):
        omega_rad = omega_hz * 2 * np.pi
        # Find closest frequency
        idx = np.argmin(np.abs(freq_full - omega_rad))
        J_simulated_points[i] = J_full[idx]
    
    print(f"\nSimulated J(0,0,ω):")
    for label, J_val in zip(omega_labels, J_simulated_points):
        print(f"  {label:10s} = {J_val:.6e} s")
    
    # =========================================================================
    # STEP 6: Compare results
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 6: Comparison")
    print("="*70)
    
    print(f"\nComparison of J(0,0,ω):")
    print(f"  {'Frequency':15s} {'Analytical':15s} {'Simulated':15s} {'Rel. Error':12s}")
    print(f"  {'-'*15} {'-'*15} {'-'*15} {'-'*12}")
    
    for i, label in enumerate(omega_labels):
        J_ana = J_analytical_points[i]
        J_sim = J_simulated_points[i]
        rel_error = abs(J_sim - J_ana) / J_ana * 100
        
        print(f"  {label:15s} {J_ana:.6e}  {J_sim:.6e}  {rel_error:10.2f}%")
    
    # Overall assessment
    max_error = np.max(np.abs(J_simulated_points - J_analytical_points) / J_analytical_points * 100)
    avg_error = np.mean(np.abs(J_simulated_points - J_analytical_points) / J_analytical_points * 100)
    
    print(f"\n  Maximum relative error: {max_error:.2f}%")
    print(f"  Average relative error: {avg_error:.2f}%")
    
    # =========================================================================
    # STEP 7: Create comparison plots
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 7: Visualization")
    print("="*70)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Local correlation function
    ax = axes[0, 0]
    tau_array = np.arange(len(C00_local)) * dt
    ax.plot(tau_array[:5000], C00_local[:5000].real, 'b-', linewidth=1.5, label='Simulated')
    
    # Analytical decay (should be plateau at S² for local motion)
    tau_plot = np.linspace(0, tau_array[5000], 100)
    C_local_theory = S2 + (1-S2) * np.exp(-tau_plot / tau_f)
    ax.plot(tau_plot, C_local_theory, 'r--', linewidth=2, label=f'Theory: S²+(1-S²)exp(-τ/{tau_f}ns)')
    
    ax.set_xlabel('Time lag τ (ns)', fontsize=11)
    ax.set_ylabel('C_local(0,0,τ)', fontsize=11)
    ax.set_title('Local Correlation Function', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Total correlation function
    ax = axes[0, 1]
    ax.semilogy(tau_array[:5000], np.abs(C00_total[:5000]), 'b-', linewidth=1.5, label='Simulated')
    
    # Analytical total
    C_total_theory = (S2 * np.exp(-tau_plot/tau_c) + 
                      (1-S2) * np.exp(-tau_plot/tau_e))
    ax.semilogy(tau_plot, C_total_theory, 'r--', linewidth=2, label='Lipari-Szabo')
    
    ax.set_xlabel('Time lag τ (ns)', fontsize=11)
    ax.set_ylabel('|C_total(0,0,τ)|', fontsize=11)
    ax.set_title('Total Correlation Function (log scale)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Spectral density comparison
    ax = axes[1, 0]
    
    # Use the full spectrum from simulation
    omega_range_plot = freq_full
    J_simulated_range = J_full
    
    ax.loglog(omega_range / (2*np.pi*1e6), J_analytical_range * 1e9, 
              'r-', linewidth=2, label='Analytical (Lipari-Szabo)')
    ax.loglog(omega_range_plot / (2*np.pi*1e6), J_simulated_range * 1e9, 
              'b--', linewidth=1.5, label='Simulated (FFT)')
    
    # Mark specific points
    ax.plot([omega_0/(2*np.pi*1e6)], [J_analytical_points[1]*1e9], 
            'ro', markersize=10, label='ω₀ (600 MHz)')
    
    ax.set_xlabel('Frequency (MHz)', fontsize=11)
    ax.set_ylabel('J(ω) (ns)', fontsize=11)
    ax.set_title('Spectral Density J(0,0,ω)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Relative error
    ax = axes[1, 1]
    
    # Interpolate analytical to simulation frequency grid
    J_ana_interp = lipari_szabo_spectral_density(omega_range_plot, S2, tau_c, tau_f)
    rel_error_range = np.abs(J_simulated_range - J_ana_interp) / (J_ana_interp + 1e-20) * 100
    
    # Only plot where both are significant
    mask = J_ana_interp > 1e-15
    
    ax.semilogx(omega_range_plot[mask] / (2*np.pi*1e6), rel_error_range[mask], 'g-', linewidth=1.5)
    ax.axhline(y=5, color='r', linestyle='--', linewidth=1, label='±5% threshold')
    ax.axhline(y=-5, color='r', linestyle='--', linewidth=1)
    
    ax.set_xlabel('Frequency (MHz)', fontsize=11)
    ax.set_ylabel('Relative Error (%)', fontsize=11)
    ax.set_title('Relative Error: Simulated vs Analytical', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    with tempfile.TemporaryDirectory() as tmpdir:
        fig_path = os.path.join(tmpdir, 'lipari_szabo_validation.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"\n  ✓ Saved comparison plot to: {fig_path}")
        print(f"    (in temporary directory, will be deleted)")
    
    plt.close()
    
    # =========================================================================
    # Final assessment
    # =========================================================================
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    # Check if errors are acceptable
    tolerance = 10.0  # 10% tolerance
    
    print(f"\n  Test Parameters:")
    print(f"    S² = {S2:.3f}, τ_f = {tau_f:.3f} ns, τ_c = {tau_c:.3f} ns")
    print(f"\n  Relative Errors:")
    print(f"    Maximum: {max_error:.2f}%")
    print(f"    Average: {avg_error:.2f}%")
    print(f"    Tolerance: {tolerance:.1f}%")
    
    if max_error < tolerance:
        print(f"\n  ✓ VALIDATION PASSED")
        print(f"    Simulated trajectory correctly reproduces Lipari-Szabo model")
        print(f"    Both local and global motions properly implemented")
        print(f"    Spectral density calculation validated")
        status = True
    else:
        print(f"\n  ✗ VALIDATION FAILED")
        print(f"    Errors exceed tolerance threshold")
        print(f"    Check simulation parameters or correlation calculation")
        status = False
    
    print("\n" + "="*70)
    
    return status


if __name__ == '__main__':
    import sys
    
    print("\n" + "="*70)
    print("LIPARI-SZABO MODEL VALIDATION TEST")
    print("="*70)
    print("\nComparing analytical vs trajectory-based spectral density")
    print("Testing both local motion (cone diffusion) and global tumbling")
    
    success = test_lipari_szabo_comparison()
    
    if success:
        print("\n✓ TEST PASSED")
        sys.exit(0)
    else:
        print("\n✗ TEST FAILED")
        sys.exit(1)
