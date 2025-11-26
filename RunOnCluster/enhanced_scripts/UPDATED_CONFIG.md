# ⚙️ Configuration Update - Dev Partition

## ✅ ALL 14 Batch Scripts Updated

### New Configuration:
- **Partition:** `dev_gpu_h100` (development/testing partition)
- **Time Limit:** `00:30:00` (30 minutes)
- **Memory:** `300G` (unchanged)
- **CPUs:** `64` (unchanged)
- **GPU:** `1` (unchanged)

---

## 📝 Why These Settings?

### Development Partition (`dev_gpu_h100`)
- ✅ **Faster queue times** - Dev partitions typically have shorter wait times
- ✅ **Quick testing** - Test scripts before long production runs
- ✅ **Debugging** - Iterate quickly to fix issues

### 30-Minute Time Limit
- ⚠️ **Warning:** Some models may not complete in 30 minutes!
- ✅ **Good for:** Quick tests, small hyperparameter searches
- ⚠️ **May timeout for:** Full 20-50 trial hyperparameter optimization

---

## 🎯 Recommended Workflow

### Step 1: Test with Dev Partition (Current Config)
```bash
# Quick test run - 30 minutes
bash submit_missing_5_models.sh
```

**Check if models complete:**
- If models finish → Great! Use these settings
- If models timeout → Increase time limit (see below)

### Step 2: If Models Timeout
Increase time to 2 hours for dev partition:
```bash
# Edit all .sh files to change:
# #SBATCH --time=00:30:00
# to:
# #SBATCH --time=02:00:00
```

### Step 3: For Production Runs
Switch to production partition with full time:
```bash
# Edit all .sh files:
# #SBATCH --partition=dev_gpu_h100  →  #SBATCH --partition=gpu_h100
# #SBATCH --time=00:30:00  →  #SBATCH --time=72:00:00
```

---

## ⏱️ Expected Completion Times

### Typical Model Runtimes (per model, 4 experiments):

| Model Type | 20 Trials | 50 Trials | Notes |
|------------|-----------|-----------|-------|
| Statistical | 10-20 min | N/A | Prophet, TBATS, SARIMAX |
| ML Models | 15-25 min | 30-45 min | XGBoost, LightGBM, CatBoost |
| DL Models | 20-30 min | 45-90 min | N-HiTS, TFT, PatchTST |
| Literature | 5-15 min | N/A | MCD30, LEAR |
| Legacy | 15-25 min | 30-60 min | Random Forest, LSTM, CNN-LSTM |

⚠️ **30-minute limit may be tight for:**
- DL models with 50 trials
- LSTM/CNN-LSTM with complex architectures
- Models with large feature sets

---

## 🔧 Quick Adjustments

### If you need more time on dev partition:

**Option 1: Edit all at once (recommended)**
```bash
cd enhanced_scripts

# Replace 30 min with 2 hours in all .sh files
sed -i 's/#SBATCH --time=00:30:00/#SBATCH --time=02:00:00/g' enhanced_*.sh
```

**Option 2: Manually edit individual files**
Change line 10 in each `.sh` file:
```bash
#SBATCH --time=00:30:00  →  #SBATCH --time=02:00:00
```

### To switch back to production partition:

```bash
# Replace dev partition with production in all .sh files
sed -i 's/#SBATCH --partition=dev_gpu_h100/#SBATCH --partition=gpu_h100/g' enhanced_*.sh
sed -i 's/#SBATCH --time=00:30:00/#SBATCH --time=72:00:00/g' enhanced_*.sh
```

---

## 📊 Current Configuration Summary

All 14 scripts now use:

```bash
#SBATCH --partition=dev_gpu_h100
#SBATCH --time=00:30:00
```

**Files updated:**
1. enhanced_02a_prophet.sh
2. enhanced_02b_tbats.sh
3. enhanced_02c_sarimax.sh
4. enhanced_03a_xgboost.sh
5. enhanced_03b_lightgbm.sh
6. enhanced_03c_catboost.sh
7. enhanced_04a_nhits.sh
8. enhanced_04b_tft.sh
9. enhanced_04c_patchtst.sh
10. enhanced_05a_mcd30.sh
11. enhanced_05b_lear.sh
12. enhanced_06a_random_forest.sh
13. enhanced_06b_lstm.sh
14. enhanced_06c_cnn_lstm.sh

---

## 🚀 Ready to Submit!

```bash
cd /home/hu/hu_hu/hu_elnaab01/projects/my_project/Bachelor_project/RunOnCluster/enhanced_scripts

# Make executable
chmod +x *.sh

# Submit with new config (dev partition, 30 min)
bash submit_missing_5_models.sh

# Monitor
squeue --me
```

---

## 💡 Tips

1. **Start with 30 min** - See which models complete
2. **Monitor closely** - Check `.out` files for progress
3. **Adjust as needed** - Increase time if models timeout
4. **Use dev for testing** - Switch to production for final runs

---

**Configuration Status: ✅ Updated - Ready to Test!**

