#!/bin/bash
# submit_all_14_models.sh
# Submit all 14 individual model scripts in parallel

echo "=" * 80
echo "Submitting ALL 14 models to run in PARALLEL"
echo "=" * 80

# Statistical Models (3)
JOB_PROPHET=$(sbatch --parsable enhanced_02a_prophet.sh)
echo "✅ Submitted Prophet: Job $JOB_PROPHET"

JOB_TBATS=$(sbatch --parsable enhanced_02b_tbats.sh)
echo "✅ Submitted TBATS: Job $JOB_TBATS"

JOB_SARIMAX=$(sbatch --parsable enhanced_02c_sarimax.sh)
echo "✅ Submitted SARIMAX: Job $JOB_SARIMAX"

# Machine Learning Models (3)
JOB_XGBOOST=$(sbatch --parsable enhanced_03a_xgboost.sh)
echo "✅ Submitted XGBoost: Job $JOB_XGBOOST"

JOB_LIGHTGBM=$(sbatch --parsable enhanced_03b_lightgbm.sh)
echo "✅ Submitted LightGBM: Job $JOB_LIGHTGBM"

JOB_CATBOOST=$(sbatch --parsable enhanced_03c_catboost.sh)
echo "✅ Submitted CatBoost: Job $JOB_CATBOOST"

# Deep Learning Models (3)
JOB_NHITS=$(sbatch --parsable enhanced_04a_nhits.sh)
echo "✅ Submitted N-HiTS: Job $JOB_NHITS"

JOB_TFT=$(sbatch --parsable enhanced_04b_tft.sh)
echo "✅ Submitted TFT: Job $JOB_TFT"

JOB_PATCHTST=$(sbatch --parsable enhanced_04c_patchtst.sh)
echo "✅ Submitted PatchTST: Job $JOB_PATCHTST"

# Literature Models (2)
JOB_MCD30=$(sbatch --parsable enhanced_05a_mcd30.sh)
echo "✅ Submitted MCD30: Job $JOB_MCD30"

JOB_LEAR=$(sbatch --parsable enhanced_05b_lear.sh)
echo "✅ Submitted LEAR: Job $JOB_LEAR"

# Legacy Models (3)
JOB_RF=$(sbatch --parsable enhanced_06a_random_forest.sh)
echo "✅ Submitted Random Forest: Job $JOB_RF"

JOB_LSTM=$(sbatch --parsable enhanced_06b_lstm.sh)
echo "✅ Submitted LSTM: Job $JOB_LSTM"

JOB_CNNLSTM=$(sbatch --parsable enhanced_06c_cnn_lstm.sh)
echo "✅ Submitted CNN-LSTM: Job $JOB_CNNLSTM"

echo ""
echo "=" * 80
echo "🎉 ALL 14 MODELS SUBMITTED SUCCESSFULLY!"
echo "=" * 80
echo ""
echo "All models are running in PARALLEL"
echo "Monitor with: squeue --me"
echo ""
echo "Job IDs:"
echo "  Statistical: $JOB_PROPHET, $JOB_TBATS, $JOB_SARIMAX"
echo "  ML:         $JOB_XGBOOST, $JOB_LIGHTGBM, $JOB_CATBOOST"
echo "  DL:         $JOB_NHITS, $JOB_TFT, $JOB_PATCHTST"
echo "  Literature: $JOB_MCD30, $JOB_LEAR"
echo "  Legacy:     $JOB_RF, $JOB_LSTM, $JOB_CNNLSTM"
echo ""
echo "Estimated completion: ~3 hours (if all run in parallel)"
echo "Check status: squeue --me"
echo "=" * 80


