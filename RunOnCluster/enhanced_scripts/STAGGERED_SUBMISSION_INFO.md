# ⏱️ Staggered Submission Strategy

## 📋 Overview

The `submit_all_14_models.sh` script now submits models in **3 staggered batches** with **2.5 hour waits** between batches.

---

## 🎯 Why Staggered Submission?

✅ **Avoid queue limits** - Some clusters limit simultaneous jobs  
✅ **Manage resources** - Prevent overwhelming the system  
✅ **Better scheduling** - Spread load over time  
✅ **Easier monitoring** - Track batches separately  

---

## 📊 Batch Breakdown

### Batch 1: First 5 Models (Statistical + ML Start)
```
1. Prophet      (Statistical)
2. TBATS        (Statistical)
3. SARIMAX      (Statistical)
4. XGBoost      (ML)
5. LightGBM     (ML)
```
**⏳ WAIT 2h 30min**

### Batch 2: Next 5 Models (ML + DL + Literature)
```
6. CatBoost     (ML)
7. N-HiTS       (DL)
8. TFT          (DL)
9. PatchTST     (DL)
10. MCD30       (Literature)
```
**⏳ WAIT 2h 30min**

### Batch 3: Final 4 Models (Literature + Legacy)
```
11. LEAR        (Literature)
12. Random Forest (Legacy)
13. LSTM        (Legacy)
14. CNN-LSTM    (Legacy)
```

---

## ⏰ Timeline

```
Time 0:00    → Submit Batch 1 (5 models)
             → Models start running
             
Time 0:00    → Script SLEEPS for 2h 30m
             → Batch 1 models running
             
Time 2:30    → Submit Batch 2 (5 models)
             → Batch 1 may still be running
             → Script SLEEPS for 2h 30m
             
Time 5:00    → Submit Batch 3 (4 models)
             → Batch 1 & 2 may still be running
             → Script COMPLETES

Time 8-10h   → All models complete (approx)
```

---

## 🚀 How to Use

### Submit All 14 in Staggered Batches:
```bash
cd /home/hu/hu_hu/hu_elnaab01/projects/my_project/Bachelor_project/RunOnCluster/enhanced_scripts

# This will take ~5 hours to complete all submissions
# (due to 2x 2.5h waits)
bash submit_all_14_models.sh

# You can safely log out - jobs will continue
# To prevent terminal from closing, use nohup:
nohup bash submit_all_14_models.sh > submission_log.txt 2>&1 &

# Check submission progress:
tail -f submission_log.txt
```

### Monitor Jobs:
```bash
# Check all your jobs
squeue --me

# Count running jobs
squeue --me | wc -l

# Check completed results
ls ../outputs/results_*.txt | wc -l
```

---

## 💡 Alternative: Submit in Background

If you don't want to keep terminal open for 5 hours:

```bash
# Run submission script in background
nohup bash submit_all_14_models.sh > submission_log.txt 2>&1 &

# Get the process ID
echo $!

# Check if still running
ps aux | grep submit_all_14_models.sh

# View submission progress
tail -f submission_log.txt

# You can safely log out now!
```

---

## 🔄 What Happens During Waits?

While the script **sleeps**:
- ✅ Previously submitted jobs **continue running**
- ✅ You can **check progress** with `squeue --me`
- ✅ You can **log out** (jobs won't stop)
- ✅ Script resumes automatically after 2.5h

---

## 📊 Expected Results

### After Batch 1 (~3 hours):
```
outputs/
├── results_prophet.txt
├── results_tbats.txt
├── results_sarimax.txt
├── results_xgboost.txt
└── results_lightgbm.txt
```

### After Batch 2 (~6 hours):
```
+ results_catboost.txt
+ results_nhits.txt
+ results_tft.txt
+ results_patchtst.txt
+ results_mcd30.txt
```

### After Batch 3 (~8-10 hours):
```
+ results_lear.txt
+ results_random_forest.txt
+ results_lstm.txt
+ results_cnn_lstm.txt
```

**Total: 14 result files**

---

## ⚙️ Customization

### Change Wait Time:

Edit `submit_all_14_models.sh`:

```bash
# Current: 2.5 hours = 9000 seconds
sleep 9000

# 1 hour
sleep 3600

# 3 hours
sleep 10800

# 30 minutes
sleep 1800
```

### Change Batch Size:

Reorganize the models between Batch 1, 2, and 3 sections.

---

## 🎯 Compare Strategies

| Strategy | Script | Submit Time | Total Time | Queue Pressure |
|----------|--------|-------------|------------|----------------|
| **Staggered** | submit_all_14_models.sh | ~5h | ~8-10h | Low ✅ |
| **All at once** | (old version) | ~1min | ~3h | High ⚠️ |
| **Missing only** | submit_missing_5_models.sh | ~1min | ~3h | Low ✅ |

---

## 📝 Notes

- **Script runtime:** ~5 hours (includes 2x 2.5h sleeps)
- **Model completion:** ~8-10 hours (script time + model runtimes)
- **You can log out** during waits - jobs continue
- **Use `nohup`** to run submission in background
- **Monitor with** `squeue --me` and `tail -f submission_log.txt`

---

## 🎉 Summary

**Old behavior:** Submit all 14 models at once  
**New behavior:** Submit in 3 batches with 2.5h delays  

**Benefit:** Better queue management, easier monitoring  
**Trade-off:** Takes ~5 hours to submit all (but you can background it)  

---

**For quick testing:** Use `submit_missing_5_models.sh` (no staggering)  
**For production:** Use `submit_all_14_models.sh` (staggered)

