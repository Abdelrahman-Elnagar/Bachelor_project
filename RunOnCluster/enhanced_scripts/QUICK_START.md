# ⚡ Quick Start Guide

## 🎯 Run Missing 5 Models (Fastest Path to Completion)

### On Cluster:

```bash
# Step 1: Navigate to folder
cd /home/hu/hu_hu/hu_elnaab01/projects/my_project/Bachelor_project/RunOnCluster/enhanced_scripts

# Step 2: Make scripts executable
chmod +x *.sh

# Step 3: Submit missing models
bash submit_missing_5_models.sh

# Step 4: Monitor
squeue --me

# Step 5: After completion (~3 hours)
python analyze_detailed_results.py
```

---

## 📋 What Gets Created

After the 5 missing models complete:

```
outputs/
├── results_tbats.txt ⭐ NEW
├── results_sarimax.txt ⭐ NEW
├── results_tft.txt ⭐ NEW
├── results_patchtst.txt ⭐ NEW
└── results_lstm.txt ⭐ NEW

detailed_results/
├── all_predictions.csv (updated with 5 new models)
├── tbats_with_wsi_solar.csv
├── tbats_no_wsi_solar.csv
├── ... (20 new files: 5 models × 4 experiments)
├── summary_statistics.csv
├── model_rankings.csv
└── experiment_winners.csv
```

---

## 🚀 Alternative: Run All 14 Models (Fresh Start)

```bash
cd /home/hu/hu_hu/hu_elnaab01/projects/my_project/Bachelor_project/RunOnCluster/enhanced_scripts

# Submit all 14 in parallel
bash submit_all_14_models.sh

# Monitor
squeue --me

# Analyze
python analyze_detailed_results.py
```

---

## ⚡ Speed Comparison

| Approach | Time | Models |
|----------|------|--------|
| Missing 5 (parallel) | ~3 hours | 5 |
| All 14 (parallel) | ~3 hours | 14 |
| All 14 (sequential) | ~35 hours | 14 |

**Parallel is 11.7x faster!** 🚀

---

## 📊 Check Progress

```bash
# Job status
squeue --me

# Completed results
ls ../outputs/results_*.txt | wc -l  # Should be 14 when done

# Detailed predictions
ls ../detailed_results/*.csv | wc -l  # Should be 56+ when done

# View a result
cat ../outputs/results_xgboost.txt
```

---

## 🎯 Your Current Status

**Have (9 models):**
- Prophet, XGBoost, LightGBM, CatBoost, N-HiTS, MCD30, LEAR, Random Forest, CNN-LSTM

**Missing (5 models):**
- TBATS, SARIMAX, TFT, PatchTST, LSTM

**One command to complete:**
```bash
bash submit_missing_5_models.sh
```

---

## 📖 Full Documentation

See `COMPLETE_GUIDE.md` for:
- Detailed usage instructions
- Troubleshooting guide
- Analysis features
- File structure
- Advanced options

---

**That's it! Run the missing 5 and you're done!** 🎉

