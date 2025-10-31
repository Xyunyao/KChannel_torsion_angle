#!/usr/bin/env python3
"""
T1 Anisotropy Analysis Script

Calculate T1 anisotropy based on CSA using diffusion on a cone model.
Supports multiple analysis tasks: chemical shift calculation, T1 calculation, and plotting.
"""

import argparse
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sympy as sp
from sympy.physics.quantum.spin import Rotation
from scipy.spatial.transform import Rotation as R
from scipy.signal import get_window
from numba import njit, prange


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Calculate T1 anisotropy based on CSA from MD trajectory orientation data'
    )
    
    # Input data (either orientation file OR simulation parameters)
    parser.add_argument('--orientation_file', type=str,
                        help='Path to NPZ file containing orientation matrices (ori_t data)')
    parser.add_argument('--chain', type=str, default='A',
                        help='Chain ID to analyze from orientation file (default: A)')
    parser.add_argument('--residue_idx', type=int, default=0,
                        help='Residue index to analyze from orientation file (default: 0, first residue)')
    
    # Diffusion on cone parameters (used only if --orientation_file not provided)
    parser.add_argument('--S2', type=float, default=0.85,
                        help='Order parameter S² for simulation (default: 0.85)')
    parser.add_argument('--tau_c', type=float, default=1e-9,
                        help='Correlation time in seconds for simulation (default: 1e-9)')
    parser.add_argument('--dt', type=float, default=5e-11,
                        help='Time step in seconds (default: 5e-11)')
    parser.add_argument('--num_steps', type=int, default=2000,
                        help='Number of simulation steps (default: 2000)')
    
    # Magnetic field
    parser.add_argument('--B0', type=float, default=14.1,
                        help='Magnetic field strength in Tesla (default: 14.1)')
    
    # CSA parameters
    parser.add_argument('--iso_val', type=float, default=170,
                        help='Isotropic chemical shift value in ppm (default: 170)')
    parser.add_argument('--delta_xx_val', type=float, default=95,
                        help='Delta xx value in ppm (default: 95)')
    parser.add_argument('--delta_yy_val', type=float, default=190,
                        help='Delta yy value in ppm (default: 190)')
    
    # Task options
    parser.add_argument('--task', type=str, default='plot_t1_vs_cs',
                        choices=['chemical_shift', 't1', 'plot_t1_vs_cs', 'ensem_t1', 'all'],
                        help='Analysis task to perform (default: plot_t1_vs_cs)')
    
    # Correlation parameters
    parser.add_argument('--max_lag', type=int, default=2000,
                        help='Maximum lag for correlation function (default: 2000). Reduce for faster computation.')
    parser.add_argument('--lag_step', type=int, default=1,
                        help='Step size for lag sampling (default: 1). WARNING: lag_step>1 reduces frequency resolution and may give inaccurate T1 values.')
    
    # Output options
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Output directory for plots and results (default: current directory)')
    parser.add_argument('--wigner_lib', type=str, 
                        default='/Users/yunyao_1/Dropbox/KcsA/analysis/lib/wigner_d_order2_N5000.npz',
                        help='Path to Wigner D-matrix library file')
    parser.add_argument('--no_show', action='store_true',
                        help='Save plots without displaying them (useful for batch processing)')
    
    return parser.parse_args()


def rotation_matrix_from_vectors(a, b):
    """Find the rotation matrix that aligns vector a to vector b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    if c == 1:
        return np.eye(3)
    if c == -1:
        # 180° rotation around arbitrary perpendicular axis
        perp = np.array([1, 0, 0]) if not np.allclose(a, [1, 0, 0]) else np.array([0, 1, 0])
        return rotation_matrix_from_vectors(a, np.cross(a, perp))
    s = np.linalg.norm(v)
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))


def load_orientation_data(orientation_file, chain='A', residue_idx=0):
    """
    Load orientation matrix data from NPZ file.
    
    Parameters
    ----------
    orientation_file : str
        Path to NPZ file containing orientation data
    chain : str
        Chain ID to extract (e.g., 'A', 'B', 'C', 'D')
    residue_idx : int
        Residue index to extract (0-based indexing)
    
    Returns
    -------
    ori_t : np.ndarray
        Orientation matrices with shape (num_frames, 1, 3, 3)
        The middle dimension is kept as 1 for compatibility
    """
    print(f"  Loading orientation data from {orientation_file}")
    print(f"  Extracting chain {chain}, residue index {residue_idx}")
    
    # Load the NPZ file
    data = np.load(orientation_file, allow_pickle=True)
    
    # Check available data
    if 'sigma_11' in data and 'sigma_22' in data and 'sigma_33' in data:
        # Data from extract_csa_orientations.py format
        sigma_11_dict = data['sigma_11'].item()
        sigma_22_dict = data['sigma_22'].item()
        sigma_33_dict = data['sigma_33'].item()
        
        if chain not in sigma_11_dict:
            raise ValueError(f"Chain {chain} not found in orientation file. Available chains: {list(sigma_11_dict.keys())}")
        
        # Get the orientation vectors for the specified chain and residue
        sigma_11 = sigma_11_dict[chain][:, residue_idx, :]  # (num_frames, 3)
        sigma_22 = sigma_22_dict[chain][:, residue_idx, :]  # (num_frames, 3)
        sigma_33 = sigma_33_dict[chain][:, residue_idx, :]  # (num_frames, 3)
        
        num_frames = sigma_11.shape[0]
        print(f"  Loaded {num_frames} frames")
        
        # Construct orientation matrices (3x3 rotation matrices)
        # Columns are [sigma_11, sigma_22, sigma_33]
        ori_t = np.zeros((num_frames, 1, 3, 3))
        ori_t[:, 0, :, 0] = sigma_11  # First column
        ori_t[:, 0, :, 1] = sigma_22  # Second column
        ori_t[:, 0, :, 2] = sigma_33  # Third column
        
        return ori_t
    
    else:
        raise ValueError(f"Unrecognized orientation file format. Expected 'sigma_11', 'sigma_22', 'sigma_33' keys.")


def simulate_vector_on_cone(S2=0.85, tau_c=0.01, dt=1e-4, num_steps=10000, axis=np.array([0, 0, 1])):
    """
    Simulate a unit vector hopping on a cone surface with fixed S2 and correlation time tau_c.
    
    Parameters:
        S2 (float): Order parameter
        tau_c (float): Correlation time in seconds
        dt (float): Time step in seconds
        num_steps (int): Number of simulation steps
        axis (np.ndarray): Cone axis direction
        
    Returns:
        vectors: (num_steps, 3) array of unit vectors
    """
    # Cone angle from S²
    cos_theta = np.sqrt((2 * S2 + 1) / 3)
    theta = np.arccos(cos_theta)

    # Ornstein-Uhlenbeck parameters for azimuthal diffusion
    gamma = 1 / tau_c
    sigma = np.sqrt(2 * gamma)  # Unit noise strength
    phi = 0.0
    axis = axis / np.linalg.norm(axis)

    # Rotation matrix to align cone with axis
    R_align = rotation_matrix_from_vectors(np.array([0, 0, 1]), axis)

    vectors = np.zeros((num_steps, 3))

    for i in range(num_steps):
        # Update azimuthal angle using Ornstein-Uhlenbeck process
        dphi = -gamma * phi * dt + sigma * np.sqrt(dt) * np.random.randn()
        phi += dphi

        # Point on cone with fixed θ and current φ
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        vec_local = np.array([x, y, z])

        # Rotate to align cone with specified axis
        vec_global = R_align @ vec_local
        vectors[i] = vec_global

    return vectors


def setup_tensor_definitions():
    """Setup symbolic tensor definitions for CSA and Dipolar interactions."""
    # Define symbolic variables for Euler angles and tensor coefficients
    alpha, beta, gamma = sp.symbols('alpha beta gamma', real=True)
    iso, delta_xx, delta_yy, delta_zz = sp.symbols('iso delta_xx delta_yy delta_zz', real=True)

    CSA_T_2m = {
        '-2': 0.5 * (delta_xx - delta_yy),
        '-1': sp.S(0),
        '0': sp.sqrt(3/2) * (delta_zz-(delta_xx + delta_yy + delta_zz)/3),
        '1': sp.S(0),
        '2': 0.5 * (delta_xx - delta_yy)
    }

    Dipolar_T_2m = {
        '-2': -sp.sqrt(3/8),
        '-1': sp.S(0),
        '0': sp.sqrt(1/2),
        '1': sp.S(0),
        '2': -sp.sqrt(3/8)
    }

    # Construct the full Wigner D-matrix for l=2
    m_values = [-2, -1, 0, 1, 2]
    D_2 = sp.zeros(5, 5)
    for i, m1 in enumerate(m_values):
        for j, m2 in enumerate(m_values):
            D_2[i, j] = sp.exp(-sp.I * m1 * alpha) * Rotation.d(2, m1, m2, beta) * sp.exp(-sp.I * m2 * gamma)

    return alpha, beta, gamma, delta_xx, delta_yy, delta_zz, CSA_T_2m, Dipolar_T_2m, D_2


def transform_tensor(T_2m, D_matrix):
    """Define the transformation of T_{2m} under the Wigner D-matrix."""
    m_values = ['-2', '-1', '0', '1', '2']
    T_2m_matrix = sp.Matrix([T_2m[m] for m in m_values])
    T_transformed = D_matrix * T_2m_matrix
    T_transformed = sp.simplify(T_transformed)
    return T_transformed


def calculate_euler_angles(ori_t, convention='ZYZ'):
    """
    Calculate Euler angles from orientation matrices.

    Parameters:
        ori_t (np.ndarray): Orientation matrices, shape (num_frames, num_residues, 3, 3).
        convention (str): Euler angle convention, e.g., 'ZYX', 'ZXZ', etc.

    Returns:
        np.ndarray: Euler angles, shape (num_frames, num_residues, 3) for yaw, pitch, roll in radians.
    """
    num_frames, num_residues = ori_t.shape[:2]
    euler_angles = np.zeros((num_frames, num_residues, 3))
    
    for frame in range(num_frames):
        for residue in range(num_residues):
            # Extract the 3x3 rotation matrix for this frame and residue
            rotation_matrix = ori_t[frame, residue]
            # Convert to Euler angles in radians
            euler_angles[frame, residue] = R.from_matrix(rotation_matrix).as_euler(convention, degrees=True) * np.pi / 180

    return euler_angles


def generate_local_axes(vectors):
    """
    Generate local orthonormal (x, y, z) axes for each vector.

    Parameters
    ----------
    vectors : np.ndarray, shape (N, n_res, 3)
        Array of position vectors (not necessarily normalized).

    Returns
    -------
    local_axes : np.ndarray, shape (N, n_res, 3, 3)
        Local orthonormal axes [x, y, z] for each vector.
        Each 3x3 block is a rotation matrix with columns (x̂, ŷ, ẑ).
    """
    vectors = np.asarray(vectors, dtype=float)
    N, n_res = vectors.shape[:2]

    # Normalize vectors → z axes
    z_axes = vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)
    z_global = np.array([0.0, 0.0, 1.0])

    x_axes = np.zeros_like(z_axes)
    y_axes = np.zeros_like(z_axes)

    for i in range(N):
        for j in range(n_res):
            z = z_axes[i, j]

            # Handle edge case: z nearly parallel to global z
            if np.allclose(np.abs(np.dot(z, z_global)), 1.0, atol=1e-6):
                ref = np.array([1.0, 0.0, 0.0])
            else:
                ref = z_global

            # Compute local x̂
            x = np.cross(ref, z)
            x_norm = np.linalg.norm(x)
            if x_norm < 1e-12:  # safety
                x = np.array([1.0, 0.0, 0.0])
            else:
                x /= x_norm

            # Compute local ŷ (right-handed)
            y = np.cross(z, x)
            y /= np.linalg.norm(y)

            x_axes[i, j] = x
            y_axes[i, j] = y

    # Stack as (N, n_res, 3, 3): columns = [x̂, ŷ, ẑ]
    local_axes = np.stack((x_axes, y_axes, z_axes), axis=2)

    return local_axes


def compute_correlation_matrix(Y_series, max_lag=1000, lag_step=1):
    """
    Compute correlation matrix for Y_series.
    
    Parameters
    ----------
    Y_series : dict
        Dictionary of time series data
    max_lag : int
        Maximum lag to compute
    lag_step : int
        Step size for lag sampling (e.g., 10 means compute every 10th lag)
        
    Returns
    -------
    corr_matrix : dict
        Correlation matrix
    """
    l = max(Y_series.keys())
    corr_matrix = {}
    for m1 in range(-l, l + 1):
        for m2 in range(-l, l + 1):
            corr = []
            y1 = Y_series[m1]
            y2 = Y_series[m2]
            for tau in range(0, max_lag, lag_step):
                val = np.mean(y1[:-tau or None] * np.conj(y2[tau:])) if tau > 0 else np.mean(y1 * np.conj(y2))
                corr.append(val)
            corr_matrix[(m1, m2)] = np.array(corr)
    return corr_matrix


@njit(parallel=True)
def rotate_all(D_all, A):
    """Rotate correlation matrix using Wigner D-matrices."""
    n = D_all.shape[0]
    n_frames = A.shape[2]
    out = np.empty((n, 5, 5, n_frames), dtype=np.complex128)

    for i in prange(n):
        D = D_all[i]
        D_H = D.T.conj()
        for t in range(n_frames):
            A_t = A[:, :, t]
            # Manual matmul since @ on complex can cause typing issues
            temp = np.zeros((5, 5), dtype=np.complex128)
            for r in range(5):
                for c in range(5):
                    s = 0j
                    for k in range(5):
                        s += D[r, k] * A_t[k, c]
                    temp[r, c] = s
            # Right multiply by D_H
            for r in range(5):
                for c in range(5):
                    s = 0j
                    for k in range(5):
                        s += temp[r, k] * D_H[k, c]
                    out[i, r, c, t] = s
    return out


def moving_average(data, window_size, axis=0):
    """
    Compute a moving average over a sliding window along a given axis.

    Parameters
    ----------
    data : np.ndarray
        Input data (1D or multi-dimensional).
    window_size : int
        Size of the moving window.
    axis : int, optional
        Axis along which to apply the moving average (default is 0).

    Returns
    -------
    np.ndarray
        Smoothed array of the same shape as input.
    """
    if window_size < 1:
        raise ValueError("window_size must be >= 1")

    data = np.asarray(data, dtype=float)
    kernel = np.ones(window_size) / window_size

    # Move the specified axis to the front, apply convolution, then restore original order
    data_swapped = np.moveaxis(data, axis, 0)
    smoothed = np.apply_along_axis(
        lambda m: np.convolve(m, kernel, mode='same'),
        axis=0,
        arr=data_swapped
    )
    return np.moveaxis(smoothed, 0, axis)


def spectral_density_nmr_no_window(acf, time_step):
    """
    Calculate spectral density for NMR relaxation from the autocorrelation function.

    Parameters
    ----------
    acf : numpy.ndarray
        Autocorrelation function (num_lags, num_residues).
    time_step : float
        Time interval between frames.

    Returns
    -------
    frequencies : numpy.ndarray
        Frequency axis (in same unit as 1/time_step).
    J : numpy.ndarray
        Spectral density (num_freqs, num_residues).
    """
    num_lags, num_residues = acf.shape

    # Zero-padding to improve frequency resolution (optional)
    n_fft = 2 * num_lags

    # FFT without windowing
    acf_mean = acf[-100:, :].mean(axis=0)
    fft_result = np.fft.rfft(acf - acf_mean, n=n_fft, axis=0)

    # Real spectral density (normalized properly)
    J = 2 * time_step * np.real(fft_result)

    # Frequencies corresponding to FFT
    frequencies = np.fft.rfftfreq(n_fft, d=time_step)

    return frequencies, J


def calculate_J_value(frequencies, J, larmor_frequency):
    """
    Calculate the spectral density J value for certain frequencies.
    
    Parameters
    ----------
    frequencies : numpy.ndarray
        Frequency array
    J : numpy.ndarray
        Spectral density (num_freqs, num_residues).
    larmor_frequency : float
        Larmor frequency in rad/s.

    Returns
    -------
    tuple
        J values at 0, omega0, and 2*omega0 for residues : 3 * num_residues
    """
    freqs = frequencies
    spe_den = J

    idx_0 = np.argmin(np.abs(freqs - 20000))  # use 20kHz instead of 0 to avoid zero frequency issue
    idx_w = np.argmin(np.abs(freqs - larmor_frequency))
    idx_2w = np.argmin(np.abs(freqs - 2 * larmor_frequency))
    
    return spe_den[idx_0], spe_den[idx_w], spe_den[idx_2w]


def calculate_T1(frequencies, spectral_density, gamma13=2 * np.pi * 10.705e6, B0=17.6):
    """
    Calculate NMR T1 relaxation time using spectral density.

    Parameters
    ----------
    frequencies : numpy.ndarray
        Array of frequencies in rad/s.
    spectral_density : numpy.ndarray
        Spectral density values corresponding to the frequencies.
    gamma13 : float
        Gyromagnetic ratio of 13C (default 2π × 10.705 × 10^6).
    B0 : float
        Magnetic field strength in T (default 17.6 T for 750MHz).

    Returns
    -------
    float
        Calculated T_1 relaxation time in seconds.
    """
    # Calculate omega0 (Larmor frequency in rad/s)
    omega0 = gamma13 * B0  # units: rad/s
    larmor_frequency = omega0 * 2 * np.pi  # in rad/s

    # J(omega0) is the spectral density at the Larmor frequency
    J_omega0 = calculate_J_value(frequencies, spectral_density, larmor_frequency)[1]

    R1 = (larmor_frequency**2) * J_omega0 * 10**(-12)  # 10**(-12) comes from ppm*ppm in correlation function

    T1 = 1.0 / R1

    return T1


def calculate_ensemble_T1(rotated_corrs, dt, gamma13=2 * np.pi * 10.705e6, B0=14.1):
    """
    Calculate ensemble-averaged T1 by averaging autocorrelation matrices first.
    
    This function averages the rotated correlation matrices over all orientations,
    then calculates a single T1 value from the averaged autocorrelation function.
    This approach differs from calculating individual T1 values and then averaging them.
    
    Parameters
    ----------
    rotated_corrs : numpy.ndarray
        Rotated correlation matrices with shape (num_orientations, 5, 5, num_lags)
    dt : float
        Time step for correlation function in seconds
    gamma13 : float
        Gyromagnetic ratio of 13C (default 2π × 10.705 × 10^6)
    B0 : float
        Magnetic field strength in Tesla (default 14.1 T)
    
    Returns
    -------
    T1_ensemble : float
        Ensemble-averaged T1 relaxation time in seconds
    frequencies : numpy.ndarray
        Frequency array from spectral density calculation
    J_ensemble : numpy.ndarray
        Ensemble-averaged spectral density
    """
    print("  Averaging autocorrelation matrices over all orientations...")
    
    # Average the correlation matrices over all orientations (axis 0)
    avg_corr_matrix = np.mean(rotated_corrs, axis=0)
    print(f"  Averaged correlation matrix shape: {avg_corr_matrix.shape}")
    
    # Extract the (1,1) component for T1 calculation
    # Index [1, 1] corresponds to m=-1 to m=-1 correlation
    avg_acf = avg_corr_matrix[1, 1, :]
    print(f"  Extracted ACF shape: {avg_acf.shape}")
    
    # Calculate spectral density from averaged ACF
    # Reshape to (num_lags, 1) for compatibility with spectral_density_nmr_no_window
    avg_acf_reshaped = avg_acf.reshape(-1, 1)
    
    # Use dt directly - it should represent the actual time between data points
    # For MD trajectories: dt = time between saved frames
    # For simulations: dt = simulation timestep
    frequencies, J_ensemble = spectral_density_nmr_no_window(avg_acf_reshaped, dt)
    
    # Smooth the spectral density
    J_ensemble_smooth = moving_average(J_ensemble, 5, axis=0)
    
    # Calculate T1 from ensemble-averaged spectral density
    T1_ensemble = calculate_T1(
        frequencies=frequencies,
        spectral_density=J_ensemble_smooth[:, 0],  # Extract single column
        gamma13=gamma13,
        B0=B0
    )
    
    print(f"  Ensemble-averaged T1: {T1_ensemble:.4f} s")
    
    return T1_ensemble, frequencies, J_ensemble_smooth[:, 0]


def main():
    """Main analysis function."""
    args = parse_arguments()
    
    # Set matplotlib backend to non-interactive if --no_show is used
    if args.no_show:
        matplotlib.use('Agg')  # Non-interactive backend
    
    print(f"Starting T1 Anisotropy Analysis")
    print(f"Parameters:")
    print(f"  B0 = {args.B0} T")
    print(f"  Task = {args.task}")
    
    # Determine data source
    if args.orientation_file:
        print(f"  Data source: MD trajectory ({args.orientation_file})")
        print(f"  Chain: {args.chain}, Residue index: {args.residue_idx}")
    else:
        print(f"  Data source: Simulated diffusion on cone")
        print(f"  S² = {args.S2}")
        print(f"  τ_c = {args.tau_c} s")
        print(f"  dt = {args.dt} s")
        print(f"  num_steps = {args.num_steps}")
    print()
    
    # Calculate CSA parameters
    delta_zz_val = 3 * args.iso_val - args.delta_xx_val - args.delta_yy_val
    delta_val = args.iso_val - delta_zz_val
    
    print(f"CSA Parameters:")
    print(f"  iso_val = {args.iso_val} ppm")
    print(f"  delta_xx_val = {args.delta_xx_val} ppm")
    print(f"  delta_yy_val = {args.delta_yy_val} ppm")
    print(f"  delta_zz_val = {delta_zz_val} ppm")
    print(f"  delta_val = {delta_val} ppm")
    print()
    
    # Step 1: Get orientation data (either from file or simulation)
    if args.orientation_file:
        print("Step 1: Loading orientation data from file...")
        ori_t = load_orientation_data(args.orientation_file, args.chain, args.residue_idx)
        num_frames = ori_t.shape[0]
        print(f"  Loaded orientation matrices: shape {ori_t.shape}")
    else:
        print("Step 1: Simulating vector trajectory on cone...")
        vecs = simulate_vector_on_cone(S2=args.S2, tau_c=args.tau_c, dt=args.dt, num_steps=args.num_steps)
        print(f"  Generated {vecs.shape[0]} vectors")
        print("  Converting vectors to orientation matrices...")
        vecs = np.expand_dims(vecs, axis=1)
        ori_t = generate_local_axes(vecs)
        num_frames = ori_t.shape[0]
    
    # Step 2: Setup tensor definitions
    print("Step 2: Setting up tensor definitions...")
    alpha, beta, gamma, delta_xx, delta_yy, delta_zz, CSA_T_2m, Dipolar_T_2m, D_2 = setup_tensor_definitions()
    
    # Step 3: Transform tensors
    print("Step 3: Transforming tensors...")
    csa_h = transform_tensor(CSA_T_2m, D_2)
    
    # Step 4: Convert orientation matrices to Euler angles
    print("Step 4: Converting orientation matrices to Euler angles...")
    euler_angles = calculate_euler_angles(ori_t)
    
    # Step 5: Calculate Y_l^m series with lambdify
    print("Step 5: Calculating spherical harmonic series...")
    CSA_Y_lm = []
    l = 2
    for m in range(-l, l+1):
        expr = csa_h[m+l]
        cas_m_expr_func = sp.lambdify((alpha, beta, gamma, delta_xx, delta_yy, delta_zz), expr, modules="numpy")
        CSA_Y_lm.append(cas_m_expr_func)
    
    CSA_Y_lm_value = {}
    for m in range(-l, l+1):
        expr_func = CSA_Y_lm[m+l]
        value_m = expr_func(
            euler_angles[:, :, 0],
            euler_angles[:, :, 1],
            euler_angles[:, :, 2],
            args.delta_xx_val,
            args.delta_yy_val,
            delta_zz_val
        )
        value_m = np.real(value_m)
        CSA_Y_lm_value[m] = value_m
    
    # Step 6: Calculate correlation matrix
    print("Step 6: Computing correlation matrix...")
    print(f"  Using max_lag={args.max_lag}, lag_step={args.lag_step}")
    csa_autocorrelation_L = compute_correlation_matrix(CSA_Y_lm_value, max_lag=args.max_lag, lag_step=args.lag_step)
    
    # Step 7: Load Wigner D-matrix library and perform ensemble averaging
    print("Step 7: Loading Wigner D-matrix library and performing ensemble averaging...")
    Wig_D2_lib = np.load(args.wigner_lib, allow_pickle=True)
    D_2_lib = Wig_D2_lib['d2_matrix']
    print(f"  Loaded Wigner library with shape {D_2_lib.shape}")
    
    A = np.array([csa_autocorrelation_L[(m1, m2)] for m1 in range(-2, 3) for m2 in range(-2, 3)])
    A = A.reshape(5, 5, -1)
    
    print("  Rotating correlation matrices...")
    rotated_corrs = rotate_all(D_2_lib, A)
    print(f"  Rotated correlations shape: {rotated_corrs.shape}")
    
    # Step 8: Calculate chemical shifts
    if args.task in ['chemical_shift', 'plot_t1_vs_cs', 'all']:
        print("Step 8: Calculating chemical shifts...")
        CSA_Y_lm_value_array = np.array([CSA_Y_lm_value[m] for m in range(-2, 3)])
        CSA_Y_lm_value_array2 = CSA_Y_lm_value_array.reshape(num_frames, 5, 1).transpose(1, 0, 2).squeeze(-1)
        rotated_csa = np.einsum('nij,jk->nik', D_2_lib, CSA_Y_lm_value_array2)
        rotated_csa_avg = np.mean(np.real(rotated_csa), axis=-1)
        
        # Plot chemical shift distribution
        plt.figure(figsize=(10, 6))
        plt.hist(rotated_csa_avg[:, 2] + args.iso_val, bins=150, color='steelblue', edgecolor='black')
        plt.title('Distribution of Chemical Shifts')
        plt.xlabel('Chemical Shift (ppm)')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(f'{args.output_dir}/chemical_shift_distribution.png', dpi=300)
        print(f"  Saved chemical shift distribution to {args.output_dir}/chemical_shift_distribution.png")
        if args.task == 'chemical_shift':
            if not args.no_show:
                plt.show()
            else:
                plt.close()
        else:
            plt.close()
    
    # Step 9: Calculate T1 values
    if args.task in ['t1', 'plot_t1_vs_cs', 'all']:
        print("Step 9: Calculating T1 relaxation times...")
        extracted_acf_m1 = rotated_corrs[:, 1, 1, :]
        
        # Adjust dt to account for lag_step
        # If we sampled every Nth lag, the effective time between correlation points is dt * N
        dt_effective = args.dt * args.lag_step
        print(f"  Using dt_effective = {dt_effective} s for spectral density calculation")
        print(f"  (dt = {args.dt} s × lag_step = {args.lag_step})")
        
        acf = extracted_acf_m1.T
        frequencies, J = spectral_density_nmr_no_window(acf, dt_effective)
        J_smooth = moving_average(J, 5, axis=0)
        
        T1_values = []
        for i in range(extracted_acf_m1.shape[0]):
            T1_i = calculate_T1(
                frequencies=frequencies,
                spectral_density=J_smooth[:, i],
                gamma13=2 * np.pi * 10.705e6,
                B0=args.B0
            )
            T1_values.append(T1_i)
        
        T1_values = np.array(T1_values)
        print(f"  Calculated {len(T1_values)} T1 values")
        print(f"  T1 range: {np.min(T1_values):.3f} - {np.max(T1_values):.3f} s")
        print(f"  Mean T1: {np.mean(T1_values):.3f} s")
        
        if args.task == 't1':
            plt.figure(figsize=(10, 6))
            plt.hist(T1_values, bins=50, color='steelblue', edgecolor='black')
            plt.title('Distribution of T1 Relaxation Times')
            plt.xlabel('T1 (s)')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(f'{args.output_dir}/t1_distribution.png', dpi=300)
            print(f"  Saved T1 distribution to {args.output_dir}/t1_distribution.png")
            if not args.no_show:
                plt.show()
            else:
                plt.close()
    
    # Step 10: Plot T1 vs Chemical Shift
    if args.task in ['plot_t1_vs_cs', 'all']:
        print("Step 10: Plotting T1 vs Chemical Shift...")
        plt.figure(figsize=(10, 6))
        plt.plot(rotated_csa_avg[:, 2] + args.iso_val, T1_values, marker='o', linestyle='', 
                 color='steelblue', alpha=0.6, markersize=4)
        plt.title('T1 Relaxation Time vs Chemical Shift')
        plt.xlabel('Isotropic Chemical Shift (ppm)')
        plt.ylabel('T1 (s)')
        plt.ylim(0, max(10, np.percentile(T1_values, 95)))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{args.output_dir}/t1_vs_chemical_shift.png', dpi=300)
        print(f"  Saved T1 vs Chemical Shift plot to {args.output_dir}/t1_vs_chemical_shift.png")
        if not args.no_show:
            plt.show()
        else:
            plt.close()
    
    # Step 11: Calculate ensemble-averaged T1
    if args.task in ['ensem_t1', 'all']:
        print("Step 11: Calculating ensemble-averaged T1...")
        dt_effective = args.dt * args.lag_step
        print(f"  Using dt_effective = {dt_effective} s (dt × lag_step)")
        
        T1_ensemble, freqs_ensemble, J_ensemble = calculate_ensemble_T1(
            rotated_corrs=rotated_corrs,
            dt=dt_effective,
            gamma13=2 * np.pi * 10.705e6,
            B0=args.B0
        )
        
        print(f"\n{'='*60}")
        print(f"Ensemble-Averaged T1 Result:")
        print(f"  T1_ensemble = {T1_ensemble:.4f} s")
        if args.task == 'all' and 'T1_values' in locals():
            print(f"  Mean of individual T1s: {np.mean(T1_values):.4f} s")
            print(f"  Difference: {abs(T1_ensemble - np.mean(T1_values)):.4f} s")
        print(f"{'='*60}\n")
        
        # Optionally plot the ensemble spectral density
        if args.task == 'ensem_t1':
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Plot spectral density
            ax1.semilogy(freqs_ensemble / (2 * np.pi * 1e6), J_ensemble)
            ax1.set_xlabel('Frequency (MHz)')
            ax1.set_ylabel('Spectral Density J(ω)')
            ax1.set_title('Ensemble-Averaged Spectral Density')
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(0, 200)
            
            # Mark the Larmor frequency
            omega0 = 2 * np.pi * 10.705e6 * args.B0
            larmor_freq_mhz = omega0 / (2 * np.pi * 1e6)
            ax1.axvline(larmor_freq_mhz, color='red', linestyle='--', label=f'ω₀ = {larmor_freq_mhz:.1f} MHz')
            ax1.legend()
            
            # Plot ensemble-averaged autocorrelation
            avg_corr_matrix = np.mean(rotated_corrs, axis=0)
            avg_acf = avg_corr_matrix[1, 1, :]
            time_axis = np.arange(len(avg_acf)) * dt_effective  # Effective time between correlation points
            
            ax2.plot(time_axis * 1e9, avg_acf / avg_acf[0])  # Normalized ACF
            ax2.set_xlabel('Time (ns)')
            ax2.set_ylabel('Normalized ACF')
            ax2.set_title('Ensemble-Averaged Autocorrelation Function')
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(0, min(time_axis[-1] * 1e9, 100))
            
            plt.tight_layout()
            plt.savefig(f'{args.output_dir}/ensemble_t1_analysis.png', dpi=300)
            print(f"  Saved ensemble T1 analysis to {args.output_dir}/ensemble_t1_analysis.png")
            if not args.no_show:
                plt.show()
            else:
                plt.close()
    
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
