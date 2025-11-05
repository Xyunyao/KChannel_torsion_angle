#!/usr/bin/env python3
"""
T1 Calculation Validation Script

Validates T1 calculation code by comparing two methods:
1. Direct T1 calculation from simulated trajectory using FFT-based spectral density
2. Analytical T1 from model-free spectral density using fitted τc and S²

Simulation setup:
- Uniaxial CSA tensor: δxx = δyy = 120 ppm, δzz = 170 ppm
- Diffusion on a cone trajectory (Lipari-Szabo model)
- Carbon-13 at user-specified magnetic field
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.spatial.transform import Rotation as R
import sys
import os

# Import functions from t1_anisotropy_analysis.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try importing, but define fallback functions if import fails
try:
    from t1_anisotropy_analysis import (
        spectral_density_nmr_no_window,
        calculate_T1,
        smooth_spectral_density_SG
    )
except ImportError as e:
    print(f"Warning: Could not import from t1_anisotropy_analysis.py: {e}")
    print("Using standalone implementations...")
    
    def spectral_density_nmr_no_window(acf, time_step):
        """Compute spectral density via FFT."""
        n_lags = len(acf)
        n_cols = acf.shape[1] if len(acf.shape) > 1 else 1
        
        fft_result = np.fft.rfft(acf, axis=0)
        frequencies = np.fft.rfftfreq(n_lags, d=time_step)
        J = 2 * time_step * np.real(fft_result)
        
        return frequencies, J
    
    def smooth_spectral_density_SG(J, frequencies, window_length=51, polyorder=3):
        """Smooth spectral density using Savitzky-Golay filter."""
        from scipy.signal import savgol_filter
        if len(J.shape) == 1:
            return savgol_filter(J, window_length, polyorder).reshape(-1, 1)
        else:
            smoothed = np.zeros_like(J)
            for i in range(J.shape[1]):
                smoothed[:, i] = savgol_filter(J[:, i], window_length, polyorder)
            return smoothed
    
    def calculate_J_value(frequencies, spectral_density, target_freq):
        """Get J value at target frequency."""
        idx = np.argmin(np.abs(frequencies - target_freq))
        return frequencies[idx], spectral_density[idx]
    
    def calculate_T1(frequencies, spectral_density, gamma13=2 * np.pi * 10.705e6, B0=17.6):
        """Calculate T1 from spectral density."""
        omega0 = gamma13 * B0
        larmor_frequency = omega0 / (2 * np.pi)
        
        J_omega0 = calculate_J_value(frequencies, spectral_density, larmor_frequency)[1]
        R1 = (larmor_frequency**2) * J_omega0 * 10**(-12)
        T1 = 1.0 / R1
        
        return T1


def generate_cone_diffusion_trajectory(S2, tau_c, dt, num_steps, cone_axis=np.array([0, 0, 1])):
    """
    Generate rotational diffusion trajectory on a cone (Lipari-Szabo model).
    
    Parameters
    ----------
    S2 : float
        Order parameter (0 < S2 < 1). Related to cone semi-angle by S2 = cos²θ(1+cosθ)/2
    tau_c : float
        Correlation time in seconds
    dt : float
        Time step in seconds
    num_steps : int
        Number of time steps
    cone_axis : np.ndarray
        Cone symmetry axis (default: z-axis)
    
    Returns
    -------
    rotations : list of scipy.spatial.transform.Rotation objects
        Rotation at each time step
    """
    # Calculate cone semi-angle from S2
    # For diffusion on a cone: S2 ≈ ((1/2)cos(theta)(1 + cos(theta)))^2
    # More accurate relation: S2 = (1/4)(1 + cos(theta))^2 * (2 - sin^2(theta))
    # Simplified: use S2 ≈ cos^2(theta) for moderate S2
    
    if S2 >= 1.0:
        theta_cone = 0.0  # No motion
    elif S2 <= 0.0:
        theta_cone = np.pi / 2  # Isotropic motion on hemisphere
    else:
        # Approximate: S2 ≈ P2(cos(theta))^2 where P2 is 2nd Legendre polynomial
        # P2(x) = (3x^2 - 1)/2
        # For simpler model: S2 = ((1+cos(theta))/2)^2
        cos_theta = 2 * np.sqrt(S2) - 1
        cos_theta = np.clip(cos_theta, -1, 1)  # Ensure valid range
        theta_cone = np.arccos(cos_theta)
    
    print(f"  Generating trajectory with:")
    print(f"    S² = {S2:.4f}")
    print(f"    τc = {tau_c*1e9:.2f} ns")
    print(f"    Cone half-angle = {np.degrees(theta_cone):.2f}°")
    print(f"    dt = {dt*1e12:.2f} ps")
    print(f"    Duration = {num_steps*dt*1e9:.2f} ns")
    
    rotations = []
    
    # Initial random orientation within cone
    phi_0 = np.random.uniform(0, 2*np.pi)
    theta_0 = np.random.uniform(0, theta_cone)
    
    # Convert to Euler angles
    alpha = phi_0
    beta = theta_0
    gamma = 0
    
    current_rotation = R.from_euler('ZYZ', [alpha, beta, gamma])
    rotations.append(current_rotation)
    
    # Rotational diffusion coefficient
    D_rot = 1 / (6 * tau_c)  # Simplified relation
    
    # Angular step size
    sigma_angle = np.sqrt(2 * D_rot * dt)
    
    for step in range(num_steps - 1):
        # Current Euler angles
        euler = current_rotation.as_euler('ZYZ')
        alpha, beta, gamma = euler
        
        # Random walk in (beta, alpha) within cone constraint
        # Beta (polar angle) - restricted to cone
        d_beta = np.random.normal(0, sigma_angle)
        beta_new = beta + d_beta
        
        # Reflect at cone boundary
        if beta_new > theta_cone:
            beta_new = 2*theta_cone - beta_new
        if beta_new < 0:
            beta_new = -beta_new
            
        # Alpha (azimuthal angle) - free diffusion
        d_alpha = np.random.normal(0, sigma_angle)
        alpha_new = alpha + d_alpha
        
        # Gamma (third Euler angle) - simplified, can add diffusion if needed
        gamma_new = gamma
        
        # Create new rotation
        current_rotation = R.from_euler('ZYZ', [alpha_new, beta_new, gamma_new])
        rotations.append(current_rotation)
    
    return rotations


def create_uniaxial_csa_tensor(delta_parallel, delta_perpendicular):
    """
    Create uniaxial CSA tensor in principal axis frame.
    
    Parameters
    ----------
    delta_parallel : float
        δ|| = δzz in ppm
    delta_perpendicular : float
        δ⊥ = δxx = δyy in ppm
    
    Returns
    -------
    tensor : np.ndarray (3x3)
        CSA tensor in principal axis frame
    """
    return np.diag([delta_perpendicular, delta_perpendicular, delta_parallel])


def rotate_csa_tensor(csa_tensor_paf, rotation):
    """
    Rotate CSA tensor from principal axis frame to lab frame.
    
    Parameters
    ----------
    csa_tensor_paf : np.ndarray (3x3)
        CSA tensor in principal axis frame
    rotation : scipy.spatial.transform.Rotation
        Rotation to apply
    
    Returns
    -------
    tensor_lab : np.ndarray (3x3)
        CSA tensor in lab frame
    """
    R_mat = rotation.as_matrix()
    return R_mat @ csa_tensor_paf @ R_mat.T


def calculate_traceless_tensor(tensor):
    """Calculate traceless part of tensor."""
    iso = np.trace(tensor) / 3
    return tensor - iso * np.eye(3)


def tensor_to_spherical_harmonics(tensor_traceless, B0_direction=np.array([0, 0, 1])):
    """
    Convert traceless tensor to spherical harmonic components.
    Calculate correlation relevant for T1 (rank-2, m=±1 components).
    
    For T1, we need: sum_m |Y_{2m}|²
    
    Parameters
    ----------
    tensor_traceless : np.ndarray (3x3)
        Traceless tensor
    B0_direction : np.ndarray
        Magnetic field direction (default: z-axis)
    
    Returns
    -------
    float
        |T_{2,-1}|² (used for autocorrelation)
    """
    # For traceless symmetric tensor aligned with B0 along z:
    # T_{2,0} ~ tensor_zz
    # T_{2,±1} ~ (tensor_xz ± i*tensor_yz)
    # T_{2,±2} ~ (tensor_xx - tensor_yy ± 2i*tensor_xy)
    
    # For T1, the relevant component is m=±1
    T_2_minus1 = tensor_traceless[0, 2] + 1j * tensor_traceless[1, 2]  # xz + i*yz
    
    return np.abs(T_2_minus1)**2


def calculate_autocorrelation(trajectory_values, max_lag=None):
    """
    Calculate autocorrelation function.
    
    Parameters
    ----------
    trajectory_values : np.ndarray
        Time series of values (real or complex)
    max_lag : int
        Maximum lag to calculate
    
    Returns
    -------
    acf : np.ndarray
        Autocorrelation function
    """
    n = len(trajectory_values)
    if max_lag is None:
        max_lag = n // 2
    
    # Normalize by value at t=0
    C0 = np.mean(np.abs(trajectory_values)**2)
    
    acf = np.zeros(max_lag)
    for lag in range(max_lag):
        if lag % 100 == 0:
            print(f"    Computing lag {lag}/{max_lag}...", end='\r')
        # Correlation: <A(t) * A(0)>
        correlations = trajectory_values[lag:] * np.conj(trajectory_values[:n-lag])
        acf[lag] = np.mean(correlations).real / C0
    
    print()  # New line after progress
    return acf


def fit_lipari_szabo_model(acf, time_array, initial_guess=(0.85, 1e-9)):
    """
    Fit autocorrelation function to Lipari-Szabo (extended model-free) model:
    C(t) = S² + (1 - S²) * exp(-t/τc)
    
    Parameters
    ----------
    acf : np.ndarray
        Autocorrelation function
    time_array : np.ndarray
        Time points (seconds)
    initial_guess : tuple
        (S2, tau_c) initial guess
    
    Returns
    -------
    S2_fit : float
        Fitted order parameter
    tau_c_fit : float
        Fitted correlation time (seconds)
    """
    def model(t, S2, tau_c):
        return S2 + (1 - S2) * np.exp(-t / tau_c)
    
    try:
        # Fit only up to 10*tau_c for better convergence
        tau_guess = initial_guess[1]
        fit_up_to = int(min(len(acf), 10 * tau_guess / (time_array[1] - time_array[0])))
        
        popt, pcov = curve_fit(
            model, 
            time_array[:fit_up_to], 
            acf[:fit_up_to],
            p0=initial_guess,
            bounds=([0, 0], [1, np.inf]),
            maxfev=10000
        )
        S2_fit, tau_c_fit = popt
        
        # Calculate fit quality
        fitted_curve = model(time_array[:fit_up_to], S2_fit, tau_c_fit)
        residuals = acf[:fit_up_to] - fitted_curve
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((acf[:fit_up_to] - np.mean(acf[:fit_up_to]))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return S2_fit, tau_c_fit, r_squared
        
    except Exception as e:
        print(f"  Warning: Fit failed with error: {e}")
        return initial_guess[0], initial_guess[1], 0.0


def analytical_spectral_density_lipari_szabo(omega, S2, tau_c):
    """
    Analytical spectral density for Lipari-Szabo model.
    
    J(ω) = (2/5) * [ S² * τc / (1 + (ωτc)²) + (1-S²) * τe / (1 + (ωτe)²) ]
    
    For simple model (no internal motion), τe = τc, so:
    J(ω) = (2/5) * τc / (1 + (ωτc)²)
    
    Parameters
    ----------
    omega : float or np.ndarray
        Angular frequency (rad/s)
    S2 : float
        Order parameter
    tau_c : float
        Correlation time (seconds)
    
    Returns
    -------
    J : float or np.ndarray
        Spectral density
    """
    return (2.0 / 5.0) * tau_c / (1 + (omega * tau_c)**2)


def calculate_T1_from_analytical_J(S2, tau_c, omega0, delta_sigma_ppm):
    """
    Calculate T1 from analytical spectral density.
    
    For CSA relaxation:
    R1 = (1/15) * (γ * B0 * Δσ)² * J(ω0)
    
    where Δσ is the CSA anisotropy.
    
    Parameters
    ----------
    S2 : float
        Order parameter
    tau_c : float
        Correlation time (seconds)
    omega0 : float
        Larmor frequency (rad/s)
    delta_sigma_ppm : float
        CSA anisotropy in ppm
    
    Returns
    -------
    T1 : float
        Relaxation time (seconds)
    """
    J_omega0 = analytical_spectral_density_lipari_szabo(omega0, S2, tau_c)
    
    # R1 = (1/15) * (ω0 * Δσ)² * J(ω0)
    # Note: Δσ in ppm needs to be scaled
    R1 = (1.0 / 15.0) * (omega0 * delta_sigma_ppm * 1e-6)**2 * J_omega0
    
    T1 = 1.0 / R1
    return T1


def main():
    """Main validation routine."""
    print("="*80)
    print("T1 CALCULATION VALIDATION")
    print("="*80)
    
    # Simulation parameters
    S2_true = 0.85  # Order parameter
    tau_c_true = 2e-9  # 2 ns correlation time
    dt = 2e-12  # 2 ps timestep
    num_steps = 20000  # 40 ns trajectory
    
    # CSA parameters (uniaxial)
    delta_perp = 120.0  # δ⊥ = δxx = δyy in ppm
    delta_para = 170.0  # δ|| = δzz in ppm
    delta_sigma = delta_para - delta_perp  # Anisotropy
    
    # NMR parameters
    B0 = 14.1  # Tesla (600 MHz for 1H)
    gamma_13C = 2 * np.pi * 10.705e6  # rad/(s*T)
    omega0 = gamma_13C * B0  # Larmor frequency in rad/s
    
    print(f"\nSimulation Parameters:")
    print(f"  True S² = {S2_true:.4f}")
    print(f"  True τc = {tau_c_true*1e9:.2f} ns")
    print(f"  Timestep dt = {dt*1e12:.2f} ps")
    print(f"  Number of steps = {num_steps}")
    print(f"  Total duration = {num_steps*dt*1e9:.2f} ns")
    print(f"\nCSA Parameters:")
    print(f"  δ⊥ (δxx=δyy) = {delta_perp:.1f} ppm")
    print(f"  δ|| (δzz) = {delta_para:.1f} ppm")
    print(f"  Δσ = {delta_sigma:.1f} ppm")
    print(f"\nNMR Parameters:")
    print(f"  B0 = {B0:.1f} T")
    print(f"  ¹³C Larmor freq = {omega0/(2*np.pi)*1e-6:.2f} MHz")
    
    # Step 1: Generate trajectory
    print(f"\n{'='*80}")
    print("STEP 1: Generate Diffusion on Cone Trajectory")
    print("="*80)
    rotations = generate_cone_diffusion_trajectory(S2_true, tau_c_true, dt, num_steps)
    
    # Step 2: Calculate CSA tensor in lab frame at each time
    print(f"\n{'='*80}")
    print("STEP 2: Calculate CSA Tensor Evolution")
    print("="*80)
    csa_paf = create_uniaxial_csa_tensor(delta_para, delta_perp)
    print(f"  CSA tensor (PAF):")
    print(f"{csa_paf}")
    
    # Calculate spherical harmonic component relevant for T1
    T2m1_squared_trajectory = []
    for i, rot in enumerate(rotations):
        if i % 1000 == 0:
            print(f"  Processing step {i}/{len(rotations)}...", end='\r')
        csa_lab = rotate_csa_tensor(csa_paf, rot)
        csa_traceless = calculate_traceless_tensor(csa_lab)
        T2m1_sq = tensor_to_spherical_harmonics(csa_traceless)
        T2m1_squared_trajectory.append(T2m1_sq)
    print()
    
    T2m1_squared_trajectory = np.array(T2m1_squared_trajectory)
    
    # Step 3: Calculate autocorrelation function
    print(f"\n{'='*80}")
    print("STEP 3: Calculate Autocorrelation Function")
    print("="*80)
    max_lag = min(5000, len(T2m1_squared_trajectory) // 2)
    print(f"  Max lag = {max_lag} steps ({max_lag*dt*1e9:.2f} ns)")
    acf = calculate_autocorrelation(T2m1_squared_trajectory, max_lag=max_lag)
    time_array = np.arange(max_lag) * dt
    
    # Step 4: Fit Lipari-Szabo model
    print(f"\n{'='*80}")
    print("STEP 4: Fit Autocorrelation to Lipari-Szabo Model")
    print("="*80)
    S2_fit, tau_c_fit, r_squared = fit_lipari_szabo_model(
        acf, time_array, 
        initial_guess=(S2_true, tau_c_true)
    )
    print(f"  Fitted S² = {S2_fit:.4f} (true: {S2_true:.4f}, error: {abs(S2_fit-S2_true)/S2_true*100:.2f}%)")
    print(f"  Fitted τc = {tau_c_fit*1e9:.2f} ns (true: {tau_c_true*1e9:.2f} ns, error: {abs(tau_c_fit-tau_c_true)/tau_c_true*100:.2f}%)")
    print(f"  R² = {r_squared:.4f}")
    
    # Step 5: Calculate T1 using METHOD 1 - Direct FFT-based spectral density
    print(f"\n{'='*80}")
    print("STEP 5: Calculate T1 - METHOD 1 (FFT-based Spectral Density)")
    print("="*80)
    
    # Prepare ACF for spectral density calculation
    acf_for_fft = acf.reshape(-1, 1) * (delta_sigma * 1e-6)**2  # Scale by Δσ² and convert ppm
    frequencies, J_fft = spectral_density_nmr_no_window(acf_for_fft, dt)
    
    # Smooth the spectral density
    J_fft_smooth = smooth_spectral_density_SG(J_fft, frequencies)
    
    # Calculate T1
    T1_method1 = calculate_T1(
        frequencies=frequencies,
        spectral_density=J_fft_smooth[:, 0],
        gamma13=gamma_13C,
        B0=B0
    )
    print(f"  T1 (Method 1 - FFT) = {T1_method1:.4f} s")
    
    # Step 6: Calculate T1 using METHOD 2 - Analytical with fitted parameters
    print(f"\n{'='*80}")
    print("STEP 6: Calculate T1 - METHOD 2 (Analytical with Fitted Parameters)")
    print("="*80)
    T1_method2 = calculate_T1_from_analytical_J(S2_fit, tau_c_fit, omega0, delta_sigma)
    print(f"  T1 (Method 2 - Analytical) = {T1_method2:.4f} s")
    
    # Step 7: Calculate T1 using METHOD 3 - Analytical with TRUE parameters
    print(f"\n{'='*80}")
    print("STEP 7: Calculate T1 - METHOD 3 (Analytical with True Parameters)")
    print("="*80)
    T1_method3 = calculate_T1_from_analytical_J(S2_true, tau_c_true, omega0, delta_sigma)
    print(f"  T1 (Method 3 - Analytical True) = {T1_method3:.4f} s")
    
    # Step 8: Compare results
    print(f"\n{'='*80}")
    print("COMPARISON OF RESULTS")
    print("="*80)
    print(f"  T1 Method 1 (FFT from trajectory):     {T1_method1:.4f} s")
    print(f"  T1 Method 2 (Analytical, fitted):      {T1_method2:.4f} s")
    print(f"  T1 Method 3 (Analytical, true params): {T1_method3:.4f} s")
    print(f"\n  Difference (1 vs 2): {abs(T1_method1 - T1_method2):.4f} s ({abs(T1_method1-T1_method2)/T1_method3*100:.2f}%)")
    print(f"  Difference (1 vs 3): {abs(T1_method1 - T1_method3):.4f} s ({abs(T1_method1-T1_method3)/T1_method3*100:.2f}%)")
    print(f"  Difference (2 vs 3): {abs(T1_method2 - T1_method3):.4f} s ({abs(T1_method2-T1_method3)/T1_method3*100:.2f}%)")
    
    # Determine validation status
    tolerance = 0.05  # 5% tolerance
    error_1vs3 = abs(T1_method1 - T1_method3) / T1_method3
    
    print(f"\n{'='*80}")
    if error_1vs3 < tolerance:
        print("✅ VALIDATION PASSED!")
        print(f"   FFT-based T1 agrees with analytical T1 within {tolerance*100:.0f}% tolerance")
    else:
        print("❌ VALIDATION FAILED!")
        print(f"   Error ({error_1vs3*100:.2f}%) exceeds {tolerance*100:.0f}% tolerance")
    print("="*80)
    
    # Step 9: Create validation plots
    print(f"\n{'='*80}")
    print("STEP 9: Generate Validation Plots")
    print("="*80)
    
    fig = plt.figure(figsize=(16, 10))
    
    # Plot 1: Autocorrelation function with fit
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(time_array*1e9, acf, 'b-', alpha=0.6, label='Simulated ACF')
    fit_curve = S2_fit + (1 - S2_fit) * np.exp(-time_array / tau_c_fit)
    ax1.plot(time_array*1e9, fit_curve, 'r--', linewidth=2, label=f'Fit: S²={S2_fit:.3f}, τc={tau_c_fit*1e9:.2f}ns')
    ax1.axhline(y=S2_fit, color='green', linestyle=':', label=f'S²={S2_fit:.3f}')
    ax1.set_xlabel('Time (ns)', fontsize=11)
    ax1.set_ylabel('C(t) / C(0)', fontsize=11)
    ax1.set_title('Autocorrelation Function', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Spectral density comparison
    ax2 = plt.subplot(2, 3, 2)
    omega_plot = frequencies * 2 * np.pi
    J_analytical = analytical_spectral_density_lipari_szabo(omega_plot, S2_fit, tau_c_fit) * (delta_sigma * 1e-6)**2
    ax2.loglog(frequencies*1e-6, J_fft_smooth[:, 0], 'b-', alpha=0.7, label='FFT-based J(ω)')
    ax2.loglog(frequencies*1e-6, J_analytical, 'r--', linewidth=2, label='Analytical J(ω)')
    ax2.axvline(x=omega0/(2*np.pi)*1e-6, color='green', linestyle=':', linewidth=2, label=f'ω₀ = {omega0/(2*np.pi)*1e-6:.1f} MHz')
    ax2.set_xlabel('Frequency (MHz)', fontsize=11)
    ax2.set_ylabel('J(ω) (s)', fontsize=11)
    ax2.set_title('Spectral Density Comparison', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, which='both')
    
    # Plot 3: T1 comparison bar chart
    ax3 = plt.subplot(2, 3, 3)
    methods = ['FFT\n(Trajectory)', 'Analytical\n(Fitted)', 'Analytical\n(True)']
    T1_values = [T1_method1, T1_method2, T1_method3]
    colors = ['blue', 'orange', 'green']
    bars = ax3.bar(methods, T1_values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax3.set_ylabel('T1 (seconds)', fontsize=11)
    ax3.set_title('T1 Comparison', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, T1_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 4: Trajectory visualization (first 1000 steps)
    ax4 = plt.subplot(2, 3, 4, projection='3d')
    n_plot = min(1000, len(rotations))
    trajectory_vectors = np.array([rot.apply([0, 0, 1]) for rot in rotations[:n_plot]])
    ax4.plot(trajectory_vectors[:, 0], trajectory_vectors[:, 1], trajectory_vectors[:, 2], 
             'b-', alpha=0.5, linewidth=0.5)
    ax4.scatter(trajectory_vectors[0, 0], trajectory_vectors[0, 1], trajectory_vectors[0, 2],
                color='green', s=100, marker='o', label='Start')
    ax4.scatter(trajectory_vectors[-1, 0], trajectory_vectors[-1, 1], trajectory_vectors[-1, 2],
                color='red', s=100, marker='s', label='End')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')
    ax4.set_title(f'Cone Diffusion Trajectory\n(first {n_plot} steps)', fontsize=12, fontweight='bold')
    ax4.legend()
    
    # Plot 5: Residuals of ACF fit
    ax5 = plt.subplot(2, 3, 5)
    fit_curve_full = S2_fit + (1 - S2_fit) * np.exp(-time_array / tau_c_fit)
    residuals = acf - fit_curve_full
    ax5.plot(time_array*1e9, residuals, 'b-', alpha=0.7)
    ax5.axhline(y=0, color='red', linestyle='--')
    ax5.set_xlabel('Time (ns)', fontsize=11)
    ax5.set_ylabel('Residuals', fontsize=11)
    ax5.set_title('Fit Residuals', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Error summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    summary_text = f"""
    VALIDATION SUMMARY
    {'='*50}
    
    Input Parameters:
      S² (true):          {S2_true:.4f}
      τc (true):          {tau_c_true*1e9:.2f} ns
      Δσ:                 {delta_sigma:.1f} ppm
      B₀:                 {B0:.1f} T
      ¹³C freq:           {omega0/(2*np.pi)*1e-6:.2f} MHz
    
    Fitted Parameters:
      S² (fitted):        {S2_fit:.4f}  ({abs(S2_fit-S2_true)/S2_true*100:.1f}% error)
      τc (fitted):        {tau_c_fit*1e9:.2f} ns  ({abs(tau_c_fit-tau_c_true)/tau_c_true*100:.1f}% error)
      R²:                 {r_squared:.4f}
    
    T1 Results:
      Method 1 (FFT):          {T1_method1:.4f} s
      Method 2 (Analytical):   {T1_method2:.4f} s
      Method 3 (True):         {T1_method3:.4f} s
    
    Validation:
      Error (1 vs 3):     {error_1vs3*100:.2f}%
      Status:             {"PASSED ✅" if error_1vs3 < tolerance else "FAILED ❌"}
    """
    ax6.text(0.1, 0.5, summary_text, transform=ax6.transAxes,
             fontsize=10, verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('t1_validation_results.pdf', dpi=300, bbox_inches='tight')
    print("  Saved: t1_validation_results.pdf")
    plt.show()
    
    print(f"\n{'='*80}")
    print("VALIDATION COMPLETE!")
    print("="*80)
    
    return {
        'T1_fft': T1_method1,
        'T1_analytical_fit': T1_method2,
        'T1_analytical_true': T1_method3,
        'S2_fit': S2_fit,
        'tau_c_fit': tau_c_fit,
        'error_percent': error_1vs3 * 100,
        'validation_passed': error_1vs3 < tolerance
    }


if __name__ == '__main__':
    results = main()
