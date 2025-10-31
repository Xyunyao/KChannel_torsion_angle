# Cluster Directory Selection Guide

## ⚠️ Important: Choose the Right Directory

Different HPC clusters use different temporary/working directories. Here's how to find and use the correct one for your cluster:

## 🔍 Common Directory Structures

### Option 1: `/scratch/$USER` (Most Common)
- Large, fast storage for temporary computational work
- Usually deleted after some period (days/weeks)
- Best for active computations
- **Check:** `ls -la /scratch/$USER`

### Option 2: `/local/$USER` (Common Alternative)
- Node-local storage (very fast)
- Usually on each compute node
- May be smaller than scratch
- **Check:** `ls -la /local/$USER`

### Option 3: `/tmp/$USER` (Common Alternative)
- Temporary directory on each node
- Often cleaned up after job completion
- May have limited space
- **Check:** `ls -la /tmp/$USER`

### Option 4: `$TMPDIR` (Environment Variable)
- Some clusters set this automatically
- Points to appropriate temporary storage
- **Check:** `echo $TMPDIR`

### Option 5: Home Directory `~` or `$HOME`
- Persistent storage
- Usually smaller quota
- Slower for I/O intensive work
- **Use only if other options unavailable**

## 🔧 How to Determine Your Cluster's Directory

### Method 1: Ask Your Cluster Documentation
```bash
# Check cluster documentation or wiki
# Common commands to find info:
man slurm
module avail
```

### Method 2: Check Available Directories
```bash
# On the cluster login node:
ls -la /scratch
ls -la /local  
ls -la /tmp
echo $TMPDIR
df -h   # See available filesystems
```

### Method 3: Ask Your Cluster Admin or Users
```bash
# Email your HPC support or ask colleagues:
# "What directory should I use for temporary computational work?"
```

### Method 4: Check Existing Job Scripts
```bash
# Look at example job scripts on your cluster:
ls /usr/share/doc/slurm*/
cat ~/.slurm/examples/*.sh   # If available
```

## 📝 Update Package for Your Cluster

Once you know your cluster's directory structure, update these files:

### In `QUICK_REFERENCE.txt`:
Replace `/local/$USER` or `/tmp/$USER` with your cluster's path

### In `README.md`:
Update the "Transfer to HPC Cluster" section

### In `CHECKLIST.md`:
Update transfer and extraction paths

### In `submit_t1_array.slurm`:
No changes needed - uses `$SLURM_SUBMIT_DIR` (current directory)

## 💡 Recommended Approach

### Best Practice:
1. **Transfer to your preferred directory**
   ```bash
   # Example for /scratch
   scp cluster_t1_analysis.tar.gz user@hpc:/scratch/$USER/
   
   # Example for /local
   scp cluster_t1_analysis.tar.gz user@hpc:/local/$USER/
   
   # Example for /tmp
   scp cluster_t1_analysis.tar.gz user@hpc:/tmp/$USER/
   ```

2. **Extract in the same directory**
   ```bash
   cd /scratch/$USER   # Or /local/$USER or /tmp/$USER
   tar -xzf cluster_t1_analysis.tar.gz
   cd cluster_t1_analysis
   ```

3. **Submit from that directory**
   ```bash
   ./submit_all.sh cpu
   ```

### Alternative: Use Environment Variable
Some clusters automatically set working directories:
```bash
# Check if your cluster sets these:
echo $SCRATCH
echo $WORK
echo $TMPDIR
echo $LOCAL_SCRATCH

# If available, use them:
cd $SCRATCH    # Or $WORK, $TMPDIR, etc.
```

## 🎯 What Directory to Choose?

### For This Analysis (99 jobs, ~127 MB data, ~500 MB results):

**Priority Order:**
1. ✅ **`/scratch/$USER`** - If available (best option)
2. ✅ **`/local/$USER`** - If scratch not available
3. ✅ **`$TMPDIR`** - If set by cluster
4. ✅ **`/tmp/$USER`** - If others not available
5. ⚠️  **`$HOME`** - Only if no other options (may be slow)

### Space Requirements:
- Input data: ~127 MB (cluster_t1_analysis.tar.gz)
- Extracted: ~150 MB
- Results: ~100-200 MB
- Logs: ~50 MB
- **Total: ~500 MB needed**

Most temporary directories should easily handle this.

## 🔄 Directory-Specific Considerations

### If Using `/scratch`:
- ✅ Large space
- ✅ Fast I/O
- ⚠️ May be auto-cleaned after N days
- 💡 **Download results promptly!**

### If Using `/local`:
- ✅ Very fast (node-local)
- ⚠️ Smaller space
- ⚠️ Only on compute nodes
- 💡 **May need to access from compute node, not login node**

### If Using `/tmp`:
- ✅ Always available
- ⚠️ May be cleaned after job
- ⚠️ Limited space
- 💡 **Download results immediately after job completes**

### If Using `$HOME`:
- ✅ Persistent
- ✅ Survives job cleanup
- ⚠️ Slower I/O
- ⚠️ Quota limits
- 💡 **Check quota with `quota -s` or `du -sh ~`**

## 🚨 Important Reminders

1. **Create directory if needed:**
   ```bash
   mkdir -p /scratch/$USER
   # Or mkdir -p /local/$USER
   # Or mkdir -p /tmp/$USER
   ```

2. **Check permissions:**
   ```bash
   ls -ld /scratch/$USER
   # Should show: drwx------ (user read/write/execute)
   ```

3. **Monitor space:**
   ```bash
   df -h /scratch/$USER
   du -sh /scratch/$USER/cluster_t1_analysis
   ```

4. **Clean up after analysis:**
   ```bash
   # After downloading results:
   cd /scratch/$USER  # Or appropriate directory
   rm -rf cluster_t1_analysis/
   rm cluster_t1_analysis.tar.gz
   ```

## 📋 Summary Commands for Your Cluster

Replace `CLUSTER_DIR` with your cluster's directory:

```bash
# Set your cluster directory
CLUSTER_DIR="/scratch/$USER"    # Adjust this!
# OR
CLUSTER_DIR="/local/$USER"
# OR  
CLUSTER_DIR="/tmp/$USER"
# OR
CLUSTER_DIR="$HOME"

# Transfer
scp cluster_t1_analysis.tar.gz user@hpc:${CLUSTER_DIR}/

# Extract
ssh user@hpc
cd ${CLUSTER_DIR}
tar -xzf cluster_t1_analysis.tar.gz
cd cluster_t1_analysis

# Submit
./submit_all.sh cpu

# Download results
scp -r user@hpc:${CLUSTER_DIR}/cluster_t1_analysis/results ./
```

---

**Still unsure?** Contact your HPC cluster support team and ask:
> "What directory should I use for temporary storage during SLURM job execution?"

They can provide the specific path for your cluster! 📧
