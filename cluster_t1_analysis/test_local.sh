#!/bin/bash
# ============================================================================
# Local Test Script - Test one residue before cluster submission
# ============================================================================
# Usage: ./test_local.sh [residue_number]
# Example: ./test_local.sh 22
# ============================================================================

RESIDUE_NUM=${1:-22}
RESIDUE_IDX=$((RESIDUE_NUM - 22))

echo "=========================================="
echo "Testing T1 Analysis Locally"
echo "=========================================="
echo "Residue: ${RESIDUE_NUM} (index ${RESIDUE_IDX})"
echo ""

# Set up paths
SCRIPT_DIR=./scripts
DATA_DIR=./data
LIB_DIR=./lib
RESULTS_DIR=./test_results

# Create test results directory
mkdir -p ${RESULTS_DIR}/residue_${RESIDUE_NUM}

# Input files
ORIENTATION_FILE=${DATA_DIR}/orientations_100p_4us.npz
WIGNER_LIB=${LIB_DIR}/wigner_d_order2_N5000.npz

# Check if files exist
if [ ! -f "${ORIENTATION_FILE}" ]; then
    echo "ERROR: Orientation file not found: ${ORIENTATION_FILE}"
    exit 1
fi

if [ ! -f "${WIGNER_LIB}" ]; then
    echo "ERROR: Wigner library not found: ${WIGNER_LIB}"
    exit 1
fi

echo "Running analysis..."
echo ""

# Run the analysis
cd ${SCRIPT_DIR}
python t1_anisotropy_analysis.py \
    --orientation_file ../${ORIENTATION_FILE} \
    --wigner_lib ../${WIGNER_LIB} \
    --chain A \
    --residue_idx ${RESIDUE_IDX} \
    --dt 1e-12 \
    --max_lag 2000 \
    --lag_step 1 \
    --B0 14.1 \
    --task ensem_t1 \
    --no_show \
    --output_dir ../${RESULTS_DIR}/residue_${RESIDUE_NUM}

EXIT_CODE=$?

cd ..

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "SUCCESS!"
    echo "=========================================="
    echo "Results saved to: ${RESULTS_DIR}/residue_${RESIDUE_NUM}/"
    echo ""
    echo "If this test worked, you can submit to the cluster with:"
    echo "  ./submit_all.sh cpu"
else
    echo ""
    echo "=========================================="
    echo "FAILED with exit code ${EXIT_CODE}"
    echo "=========================================="
    echo "Fix the errors before submitting to cluster"
fi

exit $EXIT_CODE
