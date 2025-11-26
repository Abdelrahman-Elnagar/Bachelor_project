#!/bin/bash
# submit_by_category.sh
# Submit models by category (choose which section to run)

echo "=" * 80
echo "SUBMIT MODELS BY CATEGORY"
echo "=" * 80
echo ""
echo "Uncomment the section you want to run:"
echo ""

# ===== STATISTICAL MODELS =====
# echo "Submitting Statistical Models..."
# sbatch enhanced_02a_prophet.sh
# sbatch enhanced_02b_tbats.sh
# sbatch enhanced_02c_sarimax.sh

# ===== MACHINE LEARNING MODELS =====
# echo "Submitting ML Models..."
# sbatch enhanced_03a_xgboost.sh
# sbatch enhanced_03b_lightgbm.sh
# sbatch enhanced_03c_catboost.sh

# ===== DEEP LEARNING MODELS =====
# echo "Submitting DL Models..."
# sbatch enhanced_04a_nhits.sh
# sbatch enhanced_04b_tft.sh
# sbatch enhanced_04c_patchtst.sh

# ===== LITERATURE MODELS =====
# echo "Submitting Literature Models..."
# sbatch enhanced_05a_mcd30.sh
# sbatch enhanced_05b_lear.sh

# ===== LEGACY MODELS =====
# echo "Submitting Legacy Models..."
# sbatch enhanced_06a_random_forest.sh
# sbatch enhanced_06b_lstm.sh
# sbatch enhanced_06c_cnn_lstm.sh

echo ""
echo "Edit this file to uncomment the sections you want to run"
echo "Then execute: bash submit_by_category.sh"


