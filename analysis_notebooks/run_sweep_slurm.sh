#!/bin/bash
#SBATCH --job-name=dipolar_scan
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err
#SBATCH --time=02:00:00        # walltime
#SBATCH --partition=compute    # change to your cluster's partition
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4      # adjust as needed
#SBATCH --mem=8G               # memory per job
#SBATCH --array=0-14           # 5 (S2) × 3 (tau_c) = 15 jobs, index 0-14

# --- PARAMETERS ---
S2_values=(0.1 0.3 0.5 0.7 1.0)
tau_c_values=(0.00001 0.0001 0.001)
dt=0.000001
num_steps=3000
B0=14.1
r=1.02e-10
gamma1=2.675e8
gamma2=2.713e7
N=40000
outdir="results"

mkdir -p "$outdir"
mkdir -p logs

# --- MAP ARRAY INDEX TO S2, tau_c ---
S2_index=$(( SLURM_ARRAY_TASK_ID / ${#tau_c_values[@]} ))
tau_index=$(( SLURM_ARRAY_TASK_ID % ${#tau_c_values[@]} ))

S2=${S2_values[$S2_index]}
tau_c=${tau_c_values[$tau_index]}

echo "Running simulation: S2=${S2}, tau_c=${tau_c}"

prefix="${outdir}/S2_${S2}_tau_${tau_c}"

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
    --output_prefix "$prefix"
