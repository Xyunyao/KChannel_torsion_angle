# 🎉 HPC Cluster T1 Analysis Package - Complete!

## 📦 Package Created Successfully

**Location:** `/Users/yunyao_1/Dropbox/KcsA/cluster_t1_analysis.tar.gz`  
**Size:** 127 MB  
**Ready for transfer to HPC cluster**

## 📁 Package Contents

```
cluster_t1_analysis/
├── 📋 README.md                    ← Start here! Complete instructions
├── ✅ CHECKLIST.md                 ← Step-by-step submission guide
├── 🚀 submit_all.sh                ← Master submission script
├── 🔬 submit_t1_array.slurm        ← SLURM job array (99 parallel jobs)
├── 📊 collect_results.slurm        ← Post-processing & plotting
├── 🧪 test_local.sh                ← Test before cluster submission
├── 📝 requirements.txt             ← Python dependencies
│
├── scripts/
│   ├── t1_anisotropy_analysis.py  ← Main analysis code
│   └── plot_ensemble_t1_vs_residue.py  ← Plotting code
│
├── data/
│   └── orientations_100p_4us.npz  ← 4μs MD trajectory (Chain A, res 22-120)
│
├── lib/
│   └── wigner_d_order2_N5000.npz  ← Wigner D-matrix library
│
├── results/                        ← Output directory (created by jobs)
└── logs/                          ← Job logs (created by jobs)
```

## 🚀 Quick Start Guide

### 1️⃣ Transfer to Cluster
```bash
# On your local machine
cd /Users/yunyao_1/Dropbox/KcsA
# Use /local/$USER or /tmp/$USER depending on your cluster
scp cluster_t1_analysis.tar.gz username@your-hpc-cluster.edu:/local/$USER/
```

### 2️⃣ Extract on Cluster
```bash
# On the HPC cluster
ssh username@your-hpc-cluster.edu
cd /local/$USER      # Or cd /tmp/$USER
tar -xzf cluster_t1_analysis.tar.gz
cd cluster_t1_analysis
```

### 3️⃣ Set Up Environment
```bash
# Load modules (adjust for your cluster)
module load anaconda3

# Create conda environment
conda create -n kcsa_torsion python=3.8
conda activate kcsa_torsion
pip install -r requirements.txt
```

### 4️⃣ Test Locally (IMPORTANT!)
```bash
./test_local.sh 22
```

### 5️⃣ Submit to Cluster
```bash
# For CPU partition
./submit_all.sh cpu

# For GPU partition  
./submit_all.sh gpu
```

## 📊 What Will Run

### Job Array (99 parallel jobs)
- **Residues:** 22-120 (Chain A)
- **Trajectory:** 4μs MD simulation (~100× more frames than test)
- **Analysis:** Full T1 anisotropy (chemical shift, T1, correlations, ensemble T1)
- **Runtime per job:** ~30-60 minutes
- **Total walltime:** ~30-60 minutes (with parallel execution)
- **Resources per job:** 1 CPU, 4 GB RAM

### Results Generated
Each residue gets:
1. `chemical_shift_distribution.png` - CSA distribution
2. `t1_distribution.png` - T1 across orientations
3. `t1_vs_chemical_shift.png` - Correlation plot
4. `ensemble_t1_analysis.png` - Spectral density & autocorrelation

Plus summary files:
- `ensemble_t1_summary.csv` - All T1 values
- `ensemble_t1_vs_residue.png` - T1 profile plot

## 🎯 Expected Timeline

1. **Transfer:** ~5-10 minutes (127 MB)
2. **Environment setup:** ~5 minutes
3. **Test run:** ~30 seconds - 2 minutes
4. **Queue wait:** Varies by cluster (minutes to hours)
5. **Job execution:** ~30-60 minutes (parallel)
6. **Post-processing:** ~1-2 minutes
7. **Total:** Typically 1-2 hours from submission to results

## 📈 Success Metrics

After completion, you should see:
```bash
# 99 completed residues
ls results/residue_*/ensemble_t1_analysis.png | wc -l
# Output: 99

# Summary CSV with 100 lines (1 header + 99 data)
wc -l results/ensemble_t1_summary.csv
# Output: 100

# Summary plot exists
ls results/ensemble_t1_vs_residue.png
```

## 🔧 Configuration Details

### SLURM Parameters
- **Partition:** fcpu (CPU) or fgpu (GPU)
- **Array size:** 22-120 (99 jobs)
- **Memory per job:** 4 GB
- **Time per job:** 30 minutes
- **Dependency:** Collection job waits for array completion

### Analysis Parameters
- **dt:** 1e-12 s (1 ps between frames)
- **max_lag:** 2000 points (2 ns correlation window)
- **lag_step:** 1 (full resolution - critical for accuracy)
- **B0:** 14.1 T (magnetic field strength)
- **Task:** all (chemical_shift, t1, plot_t1_vs_cs, ensem_t1)

## 📚 Documentation

1. **README.md** - Complete usage instructions, troubleshooting, and theory
2. **CHECKLIST.md** - Step-by-step submission checklist
3. **In-code comments** - Detailed explanations in all scripts

## 🛠️ Customization

### To analyze different chains:
Edit `submit_t1_array.slurm`, line ~45:
```bash
CHAIN="A"  # Change to B, C, or D
```

### To analyze different residue ranges:
Edit `submit_t1_array.slurm`, line ~6:
```bash
#SBATCH --array=22-120  # Change range as needed
```

### To adjust analysis parameters:
Edit `submit_t1_array.slurm`, lines ~50-53:
```bash
DT="1e-12"          # Time step
MAX_LAG="2000"      # Correlation length
LAG_STEP="1"        # WARNING: Keep at 1 for accuracy
B0="14.1"           # Magnetic field
```

## 🆘 Troubleshooting

### Common Issues

1. **Module not found:** Adjust module load commands in SLURM scripts
2. **Conda activation fails:** Try different activation method (see CHECKLIST.md)
3. **Jobs timeout:** Increase time limit in `#SBATCH --time=`
4. **Out of memory:** Increase `#SBATCH --mem=`
5. **Some residues fail:** Check logs, may have missing atoms

### Getting Help

1. Check `logs/t1_res_XX.err` for error messages
2. Run `./test_local.sh 22` to test locally
3. Review README.md troubleshooting section
4. Check your cluster's SLURM documentation

## 📞 Support Files

All scripts include:
- Detailed comments
- Error checking
- Progress reporting
- Automatic log file generation

## 🎓 Scientific Background

This analysis calculates **13C T1 relaxation times** from MD trajectory data:

1. **Extracts CSA tensor orientations** from backbone carbonyls
2. **Calculates autocorrelation functions** of tensor dynamics
3. **Computes spectral density** via Fourier transform
4. **Derives T1** from spectral density at Larmor frequency
5. **Ensemble averages** over ~5000 molecular orientations

Results show which residues have fast/slow backbone dynamics!

## ✅ Pre-Flight Checklist

Before submitting:
- [ ] Package transferred to cluster
- [ ] Extracted: `tar -xzf cluster_t1_analysis.tar.gz`
- [ ] Environment created: `conda create -n kcsa_torsion python=3.8`
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] SLURM scripts customized for your cluster
- [ ] Test run successful: `./test_local.sh 22`
- [ ] Ready to submit: `./submit_all.sh cpu`

## 🎊 You're All Set!

Everything is ready for HPC cluster submission. Follow the steps in **CHECKLIST.md** for a smooth experience.

**Good luck with your analysis!** 🚀

---

**Questions?** Check README.md or CHECKLIST.md for detailed instructions.

**Package created:** October 29, 2025  
**Analysis:** T1 Anisotropy from 4μs MD Trajectory  
**System:** KcsA ion channel, Chain A, Residues 22-120
