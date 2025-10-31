# T1 Anisotropy Analysis - HPC Cluster Submission Package

This package contains everything needed to run T1 relaxation analysis on an HPC cluster with SLURM.

## 📦 Package Contents

```
cluster_t1_analysis/
├── submit_all.sh                    # Master submission script
├── submit_t1_array.slurm           # SLURM job array script (99 parallel jobs)
├── collect_results.slurm           # Post-processing script
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── scripts/
│   ├── t1_anisotropy_analysis.py   # Main analysis script
│   └── plot_ensemble_t1_vs_residue.py  # Plotting script
├── lib/
│   └── wigner_d_order2_N5000.npz   # Wigner D-matrix library
├── data/
│   └── orientations_100p_4us.npz   # Orientation data from MD trajectory
├── results/                         # Output directory (created by jobs)
└── logs/                           # Job logs (created by jobs)
```

## 🚀 Quick Start

### 1. Transfer to HPC Cluster

```bash
# On your local machine
cd /Users/yunyao_1/Dropbox/KcsA
tar -czf cluster_t1_analysis.tar.gz cluster_t1_analysis/
scp cluster_t1_analysis.tar.gz username@hpc-login-node:/local/$USER/
# OR use /tmp/$USER/ if your cluster uses tmp instead

# On the HPC cluster
cd /local/$USER/    # Or cd /tmp/$USER/
tar -xzf cluster_t1_analysis.tar.gz
cd cluster_t1_analysis
```

### 2. Set Up Environment on Cluster

```bash
# Load modules (adjust for your cluster)
module load anaconda3  # or python/3.8

# Create and activate conda environment
conda create -n kcsa_torsion python=3.8
conda activate kcsa_torsion

# Install dependencies
pip install -r requirements.txt

# Test that everything works
python scripts/t1_anisotropy_analysis.py --help
```

### 3. Configure SLURM Scripts

Edit `submit_t1_array.slurm` to adjust:

```bash
#SBATCH --partition=fcpu          # Your cluster's CPU partition
#SBATCH --mem=4G                  # Memory per job
#SBATCH --time=00:30:00           # Time limit per job

# Adjust the conda activation method for your cluster:
# Option 1: If conda is already initialized
source ~/.bashrc
conda activate kcsa_torsion

# Option 2: If using module-based conda
# module load anaconda3
# source activate kcsa_torsion

# Option 3: If conda needs initialization
# eval "$(conda shell.bash hook)"
# conda activate kcsa_torsion
```

### 4. Submit Jobs

```bash
# Make scripts executable
chmod +x submit_all.sh

# Submit to CPU partition (default)
./submit_all.sh cpu

# OR submit to GPU partition
./submit_all.sh gpu
```

## 📊 What the Analysis Does

The analysis runs in parallel with **99 jobs** (one per residue, 22-120):

1. **Job Array** (`submit_t1_array.slurm`):
   - Each job processes one residue
   - Calculates T1 relaxation from CSA orientations
   - Performs all tasks: chemical shift, T1, correlations, ensemble T1
   - Takes ~5-30 minutes per residue (depending on cluster performance)
   - Saves results to `results/residue_XX/`

2. **Collection Job** (`collect_results.slurm`):
   - Runs automatically after all array jobs complete
   - Collects T1 values from all residues
   - Creates summary CSV file
   - Generates publication-quality plot

## 📈 Monitoring Jobs

### Check job status
```bash
# View all your jobs
squeue -u $USER

# View specific job array
squeue -j <JOB_ID>

# View job details
scontrol show job <JOB_ID>

# Count running jobs
squeue -u $USER -t RUNNING | wc -l
```

### Monitor progress in real-time
```bash
# Watch a specific residue log
tail -f logs/t1_res_22.out

# Watch collection job
tail -f logs/collect_results.out

# Check for errors
grep -i error logs/*.err
```

### Check completion
```bash
# Count completed jobs
ls results/residue_*/ensemble_t1_analysis.png | wc -l

# Should be 99 when all jobs complete
```

## 🛑 Cancel Jobs

```bash
# Cancel specific job array
scancel <ARRAY_JOB_ID>

# Cancel collection job
scancel <COLLECT_JOB_ID>

# Cancel all your jobs
scancel -u $USER
```

## 📁 Output Files

### Individual Residue Results
Each residue gets its own directory: `results/residue_XX/`
- `chemical_shift_distribution.png` - Chemical shift histogram
- `t1_distribution.png` - T1 distribution across orientations
- `t1_vs_chemical_shift.png` - T1 vs chemical shift correlation
- `ensemble_t1_analysis.png` - Ensemble-averaged spectral density and ACF

### Summary Results
- `results/ensemble_t1_summary.csv` - All residue T1 values in CSV format
- `results/ensemble_t1_vs_residue.png` - Plot of T1 vs residue number

### Job Logs
- `logs/t1_res_XX.out` - Standard output for each residue
- `logs/t1_res_XX.err` - Error output for each residue
- `logs/collect_results.out` - Collection job output

## ⚙️ Configuration Parameters

Key parameters in `submit_t1_array.slurm`:

```bash
DT="1e-12"          # Time between frames (1 ps)
MAX_LAG="2000"      # Correlation length (2000 points)
LAG_STEP="1"        # Lag sampling (1 = full resolution, DO NOT CHANGE)
B0="14.1"           # Magnetic field (14.1 Tesla)
CHAIN="A"           # Chain to analyze
```

## 🔧 Troubleshooting

### Jobs fail immediately
- Check conda environment is activated correctly
- Verify file paths in SLURM scripts
- Check `logs/*.err` files for error messages

### Jobs run out of memory
- Increase `#SBATCH --mem=4G` to higher value (e.g., 8G)

### Jobs timeout
- Increase `#SBATCH --time=00:30:00` to longer time (e.g., 01:00:00)

### Some residues fail
- Check individual log files: `logs/t1_res_XX.err`
- May be missing atoms in trajectory for those residues

### Collection job doesn't start
- Wait for all array jobs to complete
- Check dependency: `squeue -j <COLLECT_JOB_ID>`

## 📊 Expected Results

From the `orientations_1p_10ns.npz` test data:
- Mean T1: ~90 seconds
- Range: 2.9 - 719 seconds
- Outliers possible at residues with restricted motion
- Analysis time: ~7.5 seconds per residue (total ~12 minutes on good CPUs)

For `orientations_100p_4us.npz` (100× more frames):
- Expect longer computation time (~30 minutes per residue)
- Total walltime: With 99 parallel jobs, should complete in ~30-60 minutes
- Better statistics with more trajectory frames

## 🎓 Citation

If you use this analysis pipeline, please cite:
- Schrödinger Desmond MD package
- NumPy, SciPy, SymPy, Numba packages
- Relevant NMR relaxation theory papers

## 📝 Notes

- Each residue is independent - perfect for parallel computing
- Job array runs 99 jobs simultaneously (cluster permitting)
- Failed residues can be resubmitted individually
- All intermediate files are saved for debugging

## 🔬 Analysis Details

**Physics:**
- Calculates 13C T1 relaxation from CSA tensor dynamics
- Uses spectral density via Fourier transform of autocorrelation
- Ensemble averages over ~5000 orientations using Wigner D-matrices
- Proper frequency resolution maintained with lag_step=1

**Performance:**
- Optimized with Numba JIT compilation
- Parallel processing via SLURM job arrays
- Scales to any number of residues

## 📞 Support

For issues:
1. Check log files in `logs/`
2. Verify conda environment setup
3. Test single residue locally first
4. Check cluster-specific module requirements

---
**Version:** 1.0  
**Date:** October 2025  
**Author:** NMR T1 Analysis Pipeline
