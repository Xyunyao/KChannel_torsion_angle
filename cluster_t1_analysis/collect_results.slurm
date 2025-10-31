#!/bin/bash
#SBATCH --job-name=t1_collect
#SBATCH --partition=fcpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/collect_results.out
#SBATCH --error=logs/collect_results.err
#SBATCH --dependency=singleton      # Wait for all array jobs to complete

# ============================================================================
# Collect and Plot T1 Results
# ============================================================================
# This script runs after all array jobs complete
# It collects T1 values and creates summary plots
# ============================================================================

echo "=========================================="
echo "Collecting results at: $(date)"
echo "=========================================="

# Activate conda environment
source ~/.bashrc
conda activate kcsa_torsion

# Set up paths
WORK_DIR=$SLURM_SUBMIT_DIR
RESULTS_DIR=${WORK_DIR}/results
SCRIPT_DIR=${WORK_DIR}/scripts

cd ${WORK_DIR}

# Initialize CSV file
CSV_FILE=${RESULTS_DIR}/ensemble_t1_summary.csv
echo "residue_number,residue_idx,chain,T1_ensemble_s" > ${CSV_FILE}

echo "Collecting T1 values from individual residue results..."

# Loop through all residues
SUCCESS_COUNT=0
FAIL_COUNT=0

for RESNUM in {22..120}; do
    RES_IDX=$((RESNUM - 22))
    RES_DIR=${RESULTS_DIR}/residue_${RESNUM}
    LOG_FILE=${WORK_DIR}/logs/t1_res_${RESNUM}.out
    
    if [ -f "${LOG_FILE}" ]; then
        # Extract T1 value from log file
        T1_VALUE=$(grep "T1_ensemble =" ${LOG_FILE} | awk '{print $3}')
        
        if [ ! -z "$T1_VALUE" ]; then
            echo "${RESNUM},${RES_IDX},A,${T1_VALUE}" >> ${CSV_FILE}
            echo "  Residue ${RESNUM}: T1 = ${T1_VALUE} s"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            echo "${RESNUM},${RES_IDX},A,NA" >> ${CSV_FILE}
            echo "  Residue ${RESNUM}: FAILED (no T1 value found)"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    else
        echo "${RESNUM},${RES_IDX},A,NA" >> ${CSV_FILE}
        echo "  Residue ${RESNUM}: FAILED (no log file)"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
done

echo ""
echo "=========================================="
echo "Collection Summary:"
echo "  Successful: ${SUCCESS_COUNT} residues"
echo "  Failed: ${FAIL_COUNT} residues"
echo "  CSV saved: ${CSV_FILE}"
echo "=========================================="

# Create summary plot
echo ""
echo "Creating summary plot..."
cd ${SCRIPT_DIR}
python plot_ensemble_t1_vs_residue.py ${CSV_FILE} ${RESULTS_DIR}/ensemble_t1_vs_residue.png

echo ""
echo "=========================================="
echo "Post-processing complete at: $(date)"
echo "=========================================="
