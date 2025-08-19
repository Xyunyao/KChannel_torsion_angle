#!/bin/bash

# Choose S2 values to sweep
S2_values=(0.1 0.3 0.5 0.7 1.0)
tau_c_values=(0.00001 )   # 100 kHz, 10 kHz, 1 kHz

# Fixed parameters (edit these as needed)
dt=0.000001
num_steps=3000
B0=14.1
r=1.02e-10                 # example distance in meters
gamma1=2.675e8             # gyromagnetic ratio for proton in rad/s/T
gamma2=2.713e7             # gyromagnetic ratio for nitrogen-15 in rad/s/T
N=30000                    # number of ensemble members


# Output directory
outdir="results_test2"
mkdir -p "$outdir"

# Loop over all combinations of S2 and tau_c
for S2 in "${S2_values[@]}"; do
  for tau_c in "${tau_c_values[@]}"; do
    prefix="${outdir}/S2_${S2}_tau_${tau_c}"
    echo "Running simulation with S2=${S2}, tau_c=${tau_c}..."

    python3 dynamics_dipolar_static.py \
      --S2 "$S2" \
      --tau_c "$tau_c" \
      --dt "$dt" \
      --num_steps "$num_steps" \
      --B0 "$B0" \
      --r "$r" \
      --gamma1 "$gamma1" \
      --gamma2 "$gamma2" \
      --N "$N" \
      --output_prefix "$prefix" \
      --window "exponential" \
      --alpha 3.0 \
      --sigma 0.4 \
      --position 0.5

    # Make sure your Python code actually saves outputs with these filenames:
    # Example inside dynamics_dipolar_static.py:
    # np.save(output_prefix + "_freq.npy", freq)
    # np.save(output_prefix + "_Ix.npy", Ix_avg)
    # np.save(output_prefix + "_Iy.npy", Iy_avg)
    # np.save(output_prefix + "_fft.npy", fft_vals)
  done
done

echo "All simulations complete. Results saved to ${outdir}/"
