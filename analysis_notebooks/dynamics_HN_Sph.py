import os
import numpy as np
import argparse
from scipy.spatial.transform import Rotation as R
import sympy as sp


def euler_from_z_to_vec(vectors):
    z_axis = np.array([0, 0, 1])
    N = vectors.shape[0]
    euler_angles = np.zeros((N, 3))
    for i, vec in enumerate(vectors):
        v = vec / np.linalg.norm(vec)
        if np.allclose(v, z_axis):
            euler_angles[i] = [0, 0, 0]
        else:
            rot_axis = np.cross(z_axis, v)
            rot_angle = np.arccos(np.clip(np.dot(z_axis, v), -1.0, 1.0))
            r = R.from_rotvec(rot_axis / np.linalg.norm(rot_axis) * rot_angle)
            euler_angles[i] = r.as_euler('zyz', degrees=False)
    return euler_angles

def xyz2T2m(vectors, T2m_expr):
    euler_angles = euler_from_z_to_vec(vectors)
    alpha_vals, beta_vals, gamma_vals = euler_angles.T
    return np.array([T2m_expr[m](alpha_vals, beta_vals, gamma_vals) for m in T2m_expr])



def simulate_vector_on_cone(S2=0.85, tau_c=0.01, dt=1e-4, num_steps=10000):
    cos_theta = np.sqrt((2 * S2 + 1) / 3)
    theta = np.arccos(cos_theta)
    gamma = 1 / tau_c
    sigma = np.sqrt(2 * gamma)

    phi = np.zeros(num_steps)
    noise = sigma * np.sqrt(dt) * np.random.randn(num_steps)
    for i in range(1, num_steps):
        phi[i] = phi[i-1] - gamma * phi[i-1] * dt + noise[i]

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.full(num_steps, np.cos(theta))
    return np.column_stack((x, y, z))

def read_wigner_matrix(N, path="."):
    filename = os.path.join(path, f"wigner_d_order2_N{N}.npz")
    data = np.load(filename)
    return data['wig_d']

# --- Step 3: Calculate Ix/Iy trajectory ---
def calculate_Ip_trajectory(frequencies, dt):
    frequencies = np.array(frequencies, dtype=float)
    delta_angles = 2 * np.pi * frequencies * dt
    cum_angles = np.cumsum(delta_angles, axis=1)
    Ix = np.cos(cum_angles)
    Iy = np.sin(cum_angles)
    Ix_avg = Ix.mean(axis=0).squeeze()
    Iy_avg = Iy.mean(axis=0).squeeze()
    return Ix_avg, Iy_avg

# --- Step 4: Calculate spectrum ---

def calculate_spectrum(Ix, dt, window='exponential', alpha=3.0, sigma=0.4, position=0.5):
    """
    Calculate spectrum using FFT with different window functions.

    Parameters
    ----------
    Ix : ndarray, shape (M,)
        Time-domain trajectory.
    dt : float
        Time step for the simulation.
    window : str, optional
        Window function type: 'hann', 'hamming', 'blackman', 'exponential', 
        'gaussian', 'sinebell', or 'none'.
    alpha : float, optional
        Decay rate for exponential window (used if window='exponential').
    sigma : float, optional
        Standard deviation factor for gaussian window (used if window='gaussian').
    position : float, optional
        Relative peak location of gaussian window in [0, 1].
        0 = start, 0.5 = center, 1 = end.

    Returns
    -------
    freq : ndarray
        Frequency axis.
    fft_vals : ndarray
        FFT of windowed Ix trajectory.
    """
    N = len(Ix)
    n = np.arange(N)

    # --- Select window ---
    if window.lower() == 'hann':
        w = np.hanning(N)
    elif window.lower() == 'hamming':
        w = np.hamming(N)
    elif window.lower() == 'blackman':
        w = np.blackman(N)
    elif window.lower() == 'exponential':
        w = np.exp(-alpha * n / N)
    elif window.lower() == 'gaussian':
        # peak index determined by position parameter
        peak_idx = position * (N - 1)
        w = np.exp(-0.5 * ((n - peak_idx) / (sigma * (N-1)/2))**2)
    elif window.lower() == 'sinebell':
        w = np.sin(np.pi * n / (N-1))
    elif window.lower() == 'none':
        w = np.ones(N)
    else:
        raise ValueError(f"Unknown window type '{window}'")

    # --- Apply window to signal ---
    Ix_win = Ix * w

    # --- FFT ---
    fft_vals = np.fft.fft(Ix_win) * dt
    freq = np.fft.fftfreq(N, d=dt)

    return freq, fft_vals

# --- main script ---
def main():
    parser = argparse.ArgumentParser(description="Simulate cone dynamics effect on dipolar spectra.")
    parser.add_argument("--S2", type=float, default=0.85,
                        help="Order parameter (default 0.85)")
    parser.add_argument("--tau_c", type=float, default=0.01,
                        help="Correlation time (default 0.01)")
    parser.add_argument("--dt", type=float, default=1e-4,
                        help="Time step (default 1e-4)")
    parser.add_argument("--num_steps", type=int, default=10000,
                        help="Number of time steps (default 10000)")
    parser.add_argument("--B0", type=float, default=14.1,
                        help="Magnetic field in Tesla (default 14.1)")
    parser.add_argument("--r", type=float, required=True,
                        help="Internuclear distance in meters")
    parser.add_argument("--gamma1", type=float, required=True,
                        help="Gyromagnetic ratio of spin1 (rad/T/s)")
    parser.add_argument("--gamma2", type=float, required=True,
                        help="Gyromagnetic ratio of spin2 (rad/T/s)")
    parser.add_argument("--plot", action="store_true",
                        help="If set, plot the spectrum.")
    parser.add_argument("--wigner_file_path", type=str, default="wigner_d_order2_N5000.npz",
                        help="Path to the Wigner matrix file (default: wigner.npy)")
    parser.add_argument("--N", type=int, default=5000,
                        help="Number of ensemble members (default 5000)")
    parser.add_argument("--output_prefix", type=str, default=None,
                        help="Prefix for saving results (optional)")

    # --- New parameters for spectrum windowing ---
    parser.add_argument("--window", type=str, default="hann",
                        choices=["none", "hann", "hamming", "blackman", 
                                 "exponential", "gaussian", "sinebell"],
                        help="Window function for FFT (default: hann)")
    parser.add_argument("--alpha", type=float, default=3.0,
                        help="Decay rate for exponential window (default: 3.0)")
    parser.add_argument("--sigma", type=float, default=0.4,
                        help="Width factor for Gaussian window (default: 0.4)")
    parser.add_argument("--position", type=float, default=0.5,
                        help="Relative peak location for Gaussian window [0..1] (default: 0.5)")

    args = parser.parse_args()

    # 1. simulate trajectory (single vector path)
    HN_vector = simulate_vector_on_cone(args.S2, args.tau_c, args.dt, args.num_steps)


        # === Precompute dipolar prefactor ===
    hbar = 1.054571817e-34  # J*s (Planck constant over 2π)


    # === Precompute dipolar prefactor using numpy ===
    D_prefactor = -(1e-7/ (4.0 * np.pi)) * (1.0 / args.r**3) * (args.gamma1 * args.gamma2) * hbar

    # === Symbolic setup for T2m lambdas ===
    alpha_sym, beta_sym, gamma_sym = sp.symbols('alpha beta gamma', real=True)

    # Define spherical tensor rank-2 rotation terms T_rot[m]  !!
    T_rot = {
        -2: sp.exp(-2*sp.I*alpha_sym)*sp.sin(beta_sym)**2,
        -1: sp.exp(-1*sp.I*alpha_sym)*sp.sin(2*beta_sym),
        0: (3*sp.cos(beta_sym)**2 - 1),
        1: sp.exp(1*sp.I*alpha_sym)*sp.sin(2*beta_sym),
        2: sp.exp(2*sp.I*alpha_sym)*sp.sin(beta_sym)**2
    }
    # Define spherical tensor rank-2 rotation terms T_rot[m]  !!

    # Pre-lambdify T2m expressions
    T2m_expr = {}
    for m in range(-2, 3):
        expr = D_prefactor * sp.sqrt(2/sp.Integer(3)) * T_rot[m]
        T2m_expr[m] = sp.lambdify((alpha_sym, beta_sym, gamma_sym), expr, modules='numpy')



    # 2. convert to frequencies
    freq_single= xyz2T2m(HN_vector,T2m_expr=T2m_expr)

    # 3. extend to ensemble
    freq_ensemble = np.einsum('ij,ik->jk', read_wigner_matrix(args.N, args.wigner_file_path), freq_single)

    # 4. calculate Ix/Iy
    Ix_avg, Iy_avg = calculate_Ip_trajectory(freq_ensemble, args.dt)

    # 5. calculate spectrum
    freq, fft_vals = calculate_spectrum(Ix_avg, args.dt)

    # 6. save outputs if requested
    if args.output_prefix:
        np.save(f"{args.output_prefix}_Ix.npy", Ix_avg)
        np.save(f"{args.output_prefix}_Iy.npy", Iy_avg)
        np.save(f"{args.output_prefix}_freq.npy", freq)
        np.save(f"{args.output_prefix}_fft.npy", fft_vals)
        print(f"Saved results to {args.output_prefix}_*.npy")

    # 7. optional plot
    if args.plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,4))
        plt.plot(freq, np.abs(fft_vals))
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Intensity")
        plt.title("Dipolar Spectrum")
        plt.grid(True)
        plt.show()

    return freq, fft_vals

if __name__ == "__main__":
    main()


