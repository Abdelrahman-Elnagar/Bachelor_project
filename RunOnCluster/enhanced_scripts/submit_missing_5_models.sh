#!/bin/bash
# submit_missing_5_models.sh
# Submit only the 5 models missing from outputs folder

echo "=" * 80
echo "Submitting 5 MISSING models to run in PARALLEL"
echo "=" * 80

# Missing models based on your outputs folder
JOB_TBATS=$(sbatch --parsable enhanced_02b_tbats.sh)
echo "✅ Submitted TBATS: Job $JOB_TBATS"

JOB_SARIMAX=$(sbatch --parsable enhanced_02c_sarimax.sh)
echo "✅ Submitted SARIMAX: Job $JOB_SARIMAX"

JOB_TFT=$(sbatch --parsable enhanced_04b_tft.sh)
echo "✅ Submitted TFT: Job $JOB_TFT"

JOB_PATCHTST=$(sbatch --parsable enhanced_04c_patchtst.sh)
echo "✅ Submitted PatchTST: Job $JOB_PATCHTST"

JOB_LSTM=$(sbatch --parsable enhanced_06b_lstm.sh)
echo "✅ Submitted LSTM: Job $JOB_LSTM"

echo ""
echo "=" * 80
echo "🎉 ALL 5 MISSING MODELS SUBMITTED SUCCESSFULLY!"
echo "=" * 80
echo ""
echo "Models running in PARALLEL:"
echo "  1. TBATS    (Job $JOB_TBATS)"
echo "  2. SARIMAX  (Job $JOB_SARIMAX)"
echo "  3. TFT      (Job $JOB_TFT)"
echo "  4. PatchTST (Job $JOB_PATCHTST)"
echo "  5. LSTM     (Job $JOB_LSTM)"
echo ""
echo "Estimated completion: ~3 hours (if all run in parallel)"
echo "Monitor with: squeue --me"
echo ""
echo "After completion, you'll have all 14 models!"
echo "=" * 80


