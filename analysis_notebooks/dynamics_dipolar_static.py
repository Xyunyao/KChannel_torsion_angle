#!/usr/bin/env python3
import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# --- utility dipolar prefactor ---
def dipolar_prefactor_Hz(r, gamma1, gamma2):
    mu0 = 4 * np.pi * 1e-7
    hbar = 1.054571817e-34
    return (mu0/(4*np.pi)) * (gamma1*gamma2*hbar)/(r**3) / (2*np.pi)  # Hz

# --- Step 1: Simulate vector trajectory on a cone ---
def simulate_vector_on_cone(S2=0.85, tau_c=0.01, dt=1e-4, num_steps=10000):
    cos_theta = np.sqrt((2 * S2 + 1) / 3)
    theta = np.arccos(cos_theta)
    gamma = 1 / tau_c
    sigma = np.sqrt(2 * gamma)
    phi = 0.0
    vectors = np.zeros((num_steps, 3))

    for i in range(num_steps):
        dphi = -gamma * phi * dt + sigma * np.sqrt(dt) * np.random.randn()
        phi += dphi
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        vectors[i] = np.array([x, y, z])
    return vectors

# --- Step 2: Convert vectors to dipolar frequencies ---
def xyz2freq_numpy(vectors, B0, r, gamma1, gamma2):
    vectors = np.array(vectors, dtype=float)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    vectors /= norms
    
    original_shape = vectors.shape[:-1]
    vectors_flat = vectors.reshape(-1, 3)
    cos_theta = vectors_flat[:, 2]
    D = dipolar_prefactor_Hz(r, gamma1, gamma2)
    frequencies_flat = D * (3 * cos_theta**2 - 1) / 2
    frequencies = frequencies_flat.reshape(original_shape + (1,))
    return frequencies

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
def calculate_spectrum(Ix, dt):
    fft_vals = np.fft.fft(Ix)
    freq = np.fft.fftfreq(len(Ix), d=dt)
    fft_vals = fft_vals * dt
    return freq, fft_vals

# --- main script ---
def main():
    parser = argparse.ArgumentParser(description="Simulate dipolar NMR trajectories and spectra.")
    parser.add_argument("--S2", type=float, default=0.85, help="Order parameter (default 0.85)")
    parser.add_argument("--tau_c", type=float, default=0.01, help="Correlation time (default 0.01)")
    parser.add_argument("--dt", type=float, default=1e-4, help="Time step (default 1e-4)")
    parser.add_argument("--num_steps", type=int, default=10000, help="Number of time steps (default 10000)")
    parser.add_argument("--B0", type=float, default=14.1, help="Magnetic field in Tesla (default 14.1)")
    parser.add_argument("--r", type=float, required=True, help="Internuclear distance in meters")
    parser.add_argument("--gamma1", type=float, required=True, help="Gyromagnetic ratio of spin1 (rad/T/s)")
    parser.add_argument("--gamma2", type=float, required=True, help="Gyromagnetic ratio of spin2 (rad/T/s)")
    parser.add_argument("--plot", action="store_true", help="If set, plot the spectrum.")
    parser.add_argument("--N", type=int, default=10000, help="Number of ensemble members (default 10000)")
    parser.add_argument("--output_prefix", type=str, default=None, help="Prefix for saving results (optional)")
    args = parser.parse_args()

    # 1. simulate trajectory (single vector path)
    vectors = simulate_vector_on_cone(args.S2, args.tau_c, args.dt, args.num_steps)
    HN_vector = vectors  # base trajectory to rotate

    # 2. create ensemble by random rotation
    rot = R.random(num=args.N)
    rot_mtx = rot.as_matrix()          # shape (N,3,3)
    ensem_vector = np.einsum('nij,mj->nmi', rot_mtx, HN_vector)

    # 3. convert to frequencies
    freq_ensemble = xyz2freq_numpy(ensem_vector, args.B0, args.r, args.gamma1, args.gamma2)

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




