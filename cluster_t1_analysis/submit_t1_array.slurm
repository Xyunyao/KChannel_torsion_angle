#!/bin/bash
#SBATCH --job-name=t1_analysis
#SBATCH --partition=fcpu          # Use fcpu for CPU or fgpu for GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G                  # Memory per job
#SBATCH --time=00:30:00           # 30 minutes per residue
#SBATCH --array=22-120            # Residue numbers 22-120 (99 jobs)
#SBATCH --output=logs/t1_res_%a.out
#SBATCH --error=logs/t1_res_%a.err

# ============================================================================
# SLURM Job Array Script for T1 Anisotropy Analysis
# ============================================================================
# This script runs T1 analysis for each residue in parallel
# Each array task processes one residue
# ============================================================================

echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on node: $(hostname)"
echo "=========================================="

# Load required modules (adjust for your cluster)
# Uncomment and modify these lines based on your HPC environment
# module load python/3.8
# module load anaconda3
# module load gcc/9.3.0

# Activate conda environment
# Option 1: If conda is already initialized
source ~/.bashrc
conda activate kcsa_torsion

# Option 2: If you need to initialize conda first
# eval "$(conda shell.bash hook)"
# conda activate kcsa_torsion

# Option 3: If using module-based conda
# module load anaconda3
# source activate kcsa_torsion

# Set up paths
WORK_DIR=$SLURM_SUBMIT_DIR
SCRIPT_DIR=${WORK_DIR}/scripts
DATA_DIR=${WORK_DIR}/data
LIB_DIR=${WORK_DIR}/lib
RESULTS_DIR=${WORK_DIR}/results

# Input parameters
ORIENTATION_FILE=${DATA_DIR}/orientations_100p_4us.npz
WIGNER_LIB=${LIB_DIR}/wigner_d_order2_N5000.npz
CHAIN="A"
RESIDUE_NUM=$SLURM_ARRAY_TASK_ID
RESIDUE_IDX=$((RESIDUE_NUM - 22))  # Convert to 0-based index

# Analysis parameters
DT="1e-12"          # 1 ps time step
MAX_LAG="2000"      # 2 ns correlation window
LAG_STEP="1"        # Full resolution (DO NOT CHANGE)
B0="14.1"           # Magnetic field (Tesla)

# Output directory for this residue
OUTPUT_DIR=${RESULTS_DIR}/residue_${RESIDUE_NUM}
mkdir -p ${OUTPUT_DIR}

echo "Processing Residue ${RESIDUE_NUM} (index ${RESIDUE_IDX})"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Change to script directory
cd ${SCRIPT_DIR}

# Run T1 analysis
python t1_anisotropy_analysis.py \
    --orientation_file ${ORIENTATION_FILE} \
    --wigner_lib ${WIGNER_LIB} \
    --chain ${CHAIN} \
    --residue_idx ${RESIDUE_IDX} \
    --dt ${DT} \
    --max_lag ${MAX_LAG} \
    --lag_step ${LAG_STEP} \
    --B0 ${B0} \
    --task all \
    --no_show \
    --output_dir ${OUTPUT_DIR}

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "SUCCESS: Residue ${RESIDUE_NUM} completed"
    echo "Job finished at: $(date)"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "ERROR: Residue ${RESIDUE_NUM} failed with exit code ${EXIT_CODE}"
    echo "Job finished at: $(date)"
    echo "=========================================="
fi

exit $EXIT_CODE
