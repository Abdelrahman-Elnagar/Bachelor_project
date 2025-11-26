#!/bin/bash
# submit_all_14_models.sh
# Submit all 14 models in STAGGERED BATCHES (5 + 5 + 4)
# With 2.5 hour wait between batches

echo "================================================================================"
echo "STAGGERED SUBMISSION: 14 models in 3 batches with 2h 30min delays"
echo "================================================================================"
echo ""

# ============================================================================
# BATCH 1: First 5 Models
# ============================================================================
echo "🚀 BATCH 1: Submitting first 5 models..."
echo "--------------------------------------------------------------------------------"

JOB_PROPHET=$(sbatch --parsable enhanced_02a_prophet.sh)
echo "✅ [1/5] Submitted Prophet: Job $JOB_PROPHET"

JOB_TBATS=$(sbatch --parsable enhanced_02b_tbats.sh)
echo "✅ [2/5] Submitted TBATS: Job $JOB_TBATS"

JOB_SARIMAX=$(sbatch --parsable enhanced_02c_sarimax.sh)
echo "✅ [3/5] Submitted SARIMAX: Job $JOB_SARIMAX"

JOB_XGBOOST=$(sbatch --parsable enhanced_03a_xgboost.sh)
echo "✅ [4/5] Submitted XGBoost: Job $JOB_XGBOOST"

JOB_LIGHTGBM=$(sbatch --parsable enhanced_03b_lightgbm.sh)
echo "✅ [5/5] Submitted LightGBM: Job $JOB_LIGHTGBM"

echo ""
echo "📊 Batch 1 Job IDs: $JOB_PROPHET, $JOB_TBATS, $JOB_SARIMAX, $JOB_XGBOOST, $JOB_LIGHTGBM"
echo ""
echo "⏳ Waiting 2 hours 30 minutes before submitting Batch 2..."
echo "   (Sleep started at: $(date))"
sleep 9000  # 2.5 hours = 150 minutes = 9000 seconds
echo "   (Sleep ended at: $(date))"
echo ""

# ============================================================================
# BATCH 2: Next 5 Models
# ============================================================================
echo "🚀 BATCH 2: Submitting next 5 models..."
echo "--------------------------------------------------------------------------------"

JOB_CATBOOST=$(sbatch --parsable enhanced_03c_catboost.sh)
echo "✅ [6/14] Submitted CatBoost: Job $JOB_CATBOOST"

JOB_NHITS=$(sbatch --parsable enhanced_04a_nhits.sh)
echo "✅ [7/14] Submitted N-HiTS: Job $JOB_NHITS"

JOB_TFT=$(sbatch --parsable enhanced_04b_tft.sh)
echo "✅ [8/14] Submitted TFT: Job $JOB_TFT"

JOB_PATCHTST=$(sbatch --parsable enhanced_04c_patchtst.sh)
echo "✅ [9/14] Submitted PatchTST: Job $JOB_PATCHTST"

JOB_MCD30=$(sbatch --parsable enhanced_05a_mcd30.sh)
echo "✅ [10/14] Submitted MCD30: Job $JOB_MCD30"

echo ""
echo "📊 Batch 2 Job IDs: $JOB_CATBOOST, $JOB_NHITS, $JOB_TFT, $JOB_PATCHTST, $JOB_MCD30"
echo ""
echo "⏳ Waiting 2 hours 30 minutes before submitting Batch 3..."
echo "   (Sleep started at: $(date))"
sleep 9000  # 2.5 hours = 150 minutes = 9000 seconds
echo "   (Sleep ended at: $(date))"
echo ""

# ============================================================================
# BATCH 3: Final 4 Models
# ============================================================================
echo "🚀 BATCH 3: Submitting final 4 models..."
echo "--------------------------------------------------------------------------------"

JOB_LEAR=$(sbatch --parsable enhanced_05b_lear.sh)
echo "✅ [11/14] Submitted LEAR: Job $JOB_LEAR"

JOB_RF=$(sbatch --parsable enhanced_06a_random_forest.sh)
echo "✅ [12/14] Submitted Random Forest: Job $JOB_RF"

JOB_LSTM=$(sbatch --parsable enhanced_06b_lstm.sh)
echo "✅ [13/14] Submitted LSTM: Job $JOB_LSTM"

JOB_CNNLSTM=$(sbatch --parsable enhanced_06c_cnn_lstm.sh)
echo "✅ [14/14] Submitted CNN-LSTM: Job $JOB_CNNLSTM"

echo ""
echo "📊 Batch 3 Job IDs: $JOB_LEAR, $JOB_RF, $JOB_LSTM, $JOB_CNNLSTM"
echo ""

# ============================================================================
# FINAL SUMMARY
# ============================================================================
echo "================================================================================"
echo "🎉 ALL 14 MODELS SUBMITTED SUCCESSFULLY IN 3 STAGGERED BATCHES!"
echo "================================================================================"
echo ""
echo "📋 SUBMISSION SUMMARY:"
echo "  Batch 1 (5 models): Prophet, TBATS, SARIMAX, XGBoost, LightGBM"
echo "  Batch 2 (5 models): CatBoost, N-HiTS, TFT, PatchTST, MCD30"
echo "  Batch 3 (4 models): LEAR, Random Forest, LSTM, CNN-LSTM"
echo ""
echo "⏱️  TIMING:"
echo "  Batch 1 → Wait 2h 30m → Batch 2 → Wait 2h 30m → Batch 3"
echo "  Total script runtime: ~5 hours (2.5h + 2.5h waits)"
echo "  Total completion: ~8-10 hours (waits + model runtimes)"
echo ""
echo "📊 ALL JOB IDs:"
echo "  Batch 1: $JOB_PROPHET, $JOB_TBATS, $JOB_SARIMAX, $JOB_XGBOOST, $JOB_LIGHTGBM"
echo "  Batch 2: $JOB_CATBOOST, $JOB_NHITS, $JOB_TFT, $JOB_PATCHTST, $JOB_MCD30"
echo "  Batch 3: $JOB_LEAR, $JOB_RF, $JOB_LSTM, $JOB_CNNLSTM"
echo ""
echo "🔍 Monitor jobs: squeue --me"
echo "📁 Check results: ls ../outputs/results_*.txt"
echo "================================================================================"


