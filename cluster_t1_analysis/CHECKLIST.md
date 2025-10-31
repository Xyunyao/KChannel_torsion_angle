# Pre-Submission Checklist for HPC Cluster

## ✅ Files Ready

### Core Scripts
- [x] `submit_all.sh` - Master submission script
- [x] `submit_t1_array.slurm` - SLURM job array (99 parallel jobs)
- [x] `collect_results.slurm` - Post-processing script
- [x] `test_local.sh` - Local testing script

### Python Scripts
- [x] `scripts/t1_anisotropy_analysis.py` - Main analysis code
- [x] `scripts/plot_ensemble_t1_vs_residue.py` - Plotting code

### Data Files
- [x] `data/orientations_100p_4us.npz` - 4μs trajectory orientations (Chain A, residues 22-120)
- [x] `lib/wigner_d_order2_N5000.npz` - Wigner D-matrix library

### Documentation
- [x] `README.md` - Complete usage instructions
- [x] `requirements.txt` - Python dependencies

## 📋 Before Transferring to Cluster

### 1. Test Locally (RECOMMENDED)
```bash
cd cluster_t1_analysis
source ~/.bashrc
conda activate kcsa_torsion
./test_local.sh 22
```

Expected output:
- Should complete in ~30 seconds - 2 minutes
- Creates `test_results/residue_22/ensemble_t1_analysis.png`
- Prints T1_ensemble value

### 2. Package for Transfer
```bash
cd /Users/yunyao_1/Dropbox/KcsA
tar -czf cluster_t1_analysis.tar.gz cluster_t1_analysis/
ls -lh cluster_t1_analysis.tar.gz
```

Expected size: ~10-50 MB (depending on orientation file size)

### 3. Transfer to Cluster
```bash
# Replace with your cluster details
# Use /local/$USER/ or /tmp/$USER/ depending on your cluster
scp cluster_t1_analysis.tar.gz username@hpc.your-institution.edu:/local/$USER/
```

## 🖥️ On the HPC Cluster

### 1. Extract Package
```bash
ssh username@hpc.your-institution.edu
cd /local/$USER      # Or cd /tmp/$USER
tar -xzf cluster_t1_analysis.tar.gz
cd cluster_t1_analysis
```

### 2. Set Up Environment
```bash
# Load required modules (adjust for your cluster)
module load anaconda3        # or python/3.8
module load gcc/9.3.0       # if needed

# Create conda environment
conda create -n kcsa_torsion python=3.8
conda activate kcsa_torsion
pip install -r requirements.txt

# Test installation
python scripts/t1_anisotropy_analysis.py --help
```

### 3. Customize SLURM Scripts

Edit `submit_t1_array.slurm`:

**A. Update partition name** (if different from fcpu/fgpu):
```bash
#SBATCH --partition=YOUR_PARTITION_NAME
```

**B. Update resource requirements** (if needed):
```bash
#SBATCH --mem=4G              # Increase if jobs fail with OOM
#SBATCH --time=00:30:00       # Increase for slower systems
```

**C. Update conda activation** (choose method that works on your cluster):
```bash
# Method 1: If conda initialized in ~/.bashrc
source ~/.bashrc
conda activate kcsa_torsion

# Method 2: If using module system
module load anaconda3
source activate kcsa_torsion

# Method 3: Manual conda initialization
eval "$(conda shell.bash hook)"
conda activate kcsa_torsion
```

**D. Add any required modules:**
```bash
module load gcc/9.3.0
module load openmpi/4.0.3
# etc.
```

### 4. Test Single Job (CRITICAL!)
```bash
# Test locally on login node (quick test)
./test_local.sh 22

# OR submit a single test job to cluster
sbatch --array=22 submit_t1_array.slurm
squeue -u $USER
# Wait for completion, check logs/t1_res_22.out
```

### 5. Submit All Jobs
```bash
# Once test job succeeds, submit all
./submit_all.sh cpu        # For CPU partition
# OR
./submit_all.sh gpu        # For GPU partition

# Note the job IDs printed
```

## 📊 Monitoring

### Check Job Status
```bash
squeue -u $USER              # All your jobs
squeue -u $USER -t RUNNING   # Only running jobs
squeue -u $USER -t PENDING   # Only pending jobs
```

### Monitor Progress
```bash
# Watch a running job
tail -f logs/t1_res_22.out

# Count completed jobs
ls results/residue_*/ensemble_t1_analysis.png | wc -l

# Check for errors
grep -i error logs/*.err | head
grep -i failed logs/*.out | head
```

### Expected Timeline
- **Job submission**: Immediate
- **Queue wait**: Depends on cluster load (minutes to hours)
- **Per-job runtime**: ~30-60 minutes (for 100p trajectory)
- **Total walltime**: ~30-60 minutes (with parallel execution)
- **Collection job**: ~1-2 minutes after array completes

## 🎯 Expected Results

### Success Indicators
- 99 log files created: `logs/t1_res_22.out` through `logs/t1_res_120.out`
- 99 result directories: `results/residue_22/` through `results/residue_120/`
- Each directory contains 4 PNG files
- Summary CSV created: `results/ensemble_t1_summary.csv`
- Summary plot created: `results/ensemble_t1_vs_residue.png`

### Success Metrics
```bash
# Should show 99
ls results/residue_*/ensemble_t1_analysis.png | wc -l

# Should show 99 data lines + 1 header
wc -l results/ensemble_t1_summary.csv

# Should show no errors (or very few)
grep -i error logs/*.err | wc -l
```

## 🛑 Troubleshooting

### If Jobs Fail Immediately
1. Check `logs/t1_res_22.err` for error messages
2. Verify conda environment: `conda list | grep numpy`
3. Check file paths in SLURM script
4. Test locally: `./test_local.sh 22`

### If Jobs Time Out
1. Increase `#SBATCH --time=01:00:00` to 2-3 hours
2. Check if trajectory is very large
3. Consider reducing `--max_lag` from 2000 to 1000

### If Jobs Run Out of Memory
1. Increase `#SBATCH --mem=8G` or higher
2. Check memory usage: `seff <JOB_ID>`

### If Some Residues Fail
1. Check individual logs: `cat logs/t1_res_XX.err`
2. May indicate missing atoms in those residues
3. Can skip failed residues (marked as NA in CSV)

## 📥 Retrieve Results

### Download Results from Cluster
```bash
# On your local machine
# Adjust path based on your cluster (/local/$USER or /tmp/$USER)
scp -r username@hpc:/local/$USER/cluster_t1_analysis/results ./
scp username@hpc:/local/$USER/cluster_t1_analysis/logs/*.out ./logs/
```

## ✨ Post-Analysis

Once results are complete:
1. Check `results/ensemble_t1_summary.csv` for T1 values
2. View `results/ensemble_t1_vs_residue.png` for trends
3. Investigate outliers (very high/low T1 values)
4. Compare with experimental data if available
5. Examine individual residue plots in `results/residue_XX/`

---

## 🎓 Quick Reference

### Essential Commands
```bash
# Submit jobs
./submit_all.sh cpu

# Monitor
squeue -u $USER

# Cancel
scancel -u $USER

# Check completion
ls results/residue_*/ensemble_t1_analysis.png | wc -l

# View results
cat results/ensemble_t1_summary.csv
```

### File Sizes
- Orientation file: ~10-100 MB (depends on trajectory length)
- Each result directory: ~1-2 MB
- Total results: ~100-200 MB
- Log files: ~10-50 MB total

### Compute Requirements
- CPU cores: 99 (one per job, run in parallel)
- Memory per job: ~4 GB
- Total walltime: ~30-60 minutes
- Disk space: ~500 MB

---

**Ready to submit?** Follow the steps above! 🚀
