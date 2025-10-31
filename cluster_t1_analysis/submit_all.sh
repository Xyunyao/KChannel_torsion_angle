#!/bin/bash
# ============================================================================
# Master Submission Script for T1 Anisotropy Analysis on HPC Cluster
# ============================================================================
# This script submits all jobs to the SLURM cluster
# Usage: ./submit_all.sh [cpu|gpu]
# ============================================================================

# Check argument
COMPUTE_TYPE=${1:-cpu}

if [ "$COMPUTE_TYPE" == "gpu" ]; then
    PARTITION="fgpu"
    echo "Using GPU partition (fgpu)"
elif [ "$COMPUTE_TYPE" == "cpu" ]; then
    PARTITION="fcpu"
    echo "Using CPU partition (fcpu)"
else
    echo "Error: Invalid argument. Use 'cpu' or 'gpu'"
    echo "Usage: ./submit_all.sh [cpu|gpu]"
    exit 1
fi

# Update partition in SLURM scripts
sed -i.bak "s/^#SBATCH --partition=.*/#SBATCH --partition=${PARTITION}/" submit_t1_array.slurm
sed -i.bak "s/^#SBATCH --partition=.*/#SBATCH --partition=${PARTITION}/" collect_results.slurm

echo "=========================================="
echo "T1 Anisotropy Analysis - Cluster Submission"
echo "=========================================="
echo "Date: $(date)"
echo "Partition: ${PARTITION}"
echo "Working directory: $(pwd)"
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# Submit the job array (99 jobs, one per residue)
echo "Submitting job array for residues 22-120..."
ARRAY_JOB_ID=$(sbatch --parsable submit_t1_array.slurm)

if [ -z "$ARRAY_JOB_ID" ]; then
    echo "ERROR: Failed to submit job array"
    exit 1
fi

echo "  Job array submitted: Job ID ${ARRAY_JOB_ID}"
echo "  Number of jobs: 99 (residues 22-120)"
echo ""

# Submit the collection job (depends on array completion)
echo "Submitting result collection job..."
COLLECT_JOB_ID=$(sbatch --parsable --dependency=afterok:${ARRAY_JOB_ID} collect_results.slurm)

if [ -z "$COLLECT_JOB_ID" ]; then
    echo "ERROR: Failed to submit collection job"
    exit 1
fi

echo "  Collection job submitted: Job ID ${COLLECT_JOB_ID}"
echo "  (Will run after all array jobs complete)"
echo ""

echo "=========================================="
echo "Jobs submitted successfully!"
echo "=========================================="
echo ""
echo "Monitor your jobs with:"
echo "  squeue -u \$USER"
echo "  squeue -j ${ARRAY_JOB_ID}"
echo ""
echo "Check job details:"
echo "  scontrol show job ${ARRAY_JOB_ID}"
echo ""
echo "View logs in real-time:"
echo "  tail -f logs/t1_res_22.out"
echo ""
echo "Cancel jobs if needed:"
echo "  scancel ${ARRAY_JOB_ID}        # Cancel array jobs"
echo "  scancel ${COLLECT_JOB_ID}      # Cancel collection job"
echo "  scancel -u \$USER              # Cancel all your jobs"
echo ""
echo "Results will be saved to:"
echo "  results/ensemble_t1_summary.csv"
echo "  results/ensemble_t1_vs_residue.png"
echo "  results/residue_*/             (individual residue results)"
echo ""
