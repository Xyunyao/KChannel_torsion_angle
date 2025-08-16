#!/bin/bash

# Choose S2 values to sweep
S2_values=(0.1 0.3 0.5 0.7 1.0)

# Fixed parameters (edit these as needed)
tau_c=0.00001
dt=0.000001
num_steps=2000 # 
B0=14.1
r=1.02e-10                 # example distance in meters
gamma1=2.675e8  # gyromagnetic ratio for proton in /s/T
# gyromagnetic ratio for nitrogen in rad/s/T
gamma2=2.713e7  # gyromagnetic ratio for nitrogen-15 in /s/T
N=8000                     # number of ensemble members

# Output directory
outdir="results"
mkdir -p $outdir

# Loop over all S2 values
for S2 in "${S2_values[@]}"; do
    prefix="${outdir}/S2_${S2}"
    echo "Running simulation with S2=${S2}..."
    
    # Run Python simulation and save numpy outputs
    python3 dynamics_dipolar_static.py \
        --S2 $S2 \
        --tau_c $tau_c \
        --dt $dt \
        --num_steps $num_steps \
        --B0 $B0 \
        --r $r \
        --gamma1 $gamma1 \
        --gamma2 $gamma2 \
        --N $N \
        --output_prefix "${prefix}"

    # If your Python code needs to save outputs, add saving code inside dynamics_dipolar_static.py
    # For example: np.save(prefix+"_freq.npy", freq), etc.
done

echo "All simulations complete. Results saved to ${outdir}/"
