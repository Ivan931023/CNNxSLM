#!/bin/bash
source ../.venv/bin/activate

# 確保輸出不被 buffer
export PYTHONUNBUFFERED=1

echo "=================================================="
echo "🚀 終極優化計畫開始 (Phase 1/3)"
echo "=================================================="
echo "啟動 Stage 1: ResNet-34 粗調訓練 (20 Epochs)..."
python train_flattop_cnn.py

echo "=================================================="
echo "🚀 終極優化計畫開始 (Phase 2/3)"
echo "=================================================="
echo "啟動 Stage 2: 生成微調資料集與完美錨點 (5100 筆)..."
python dataset_generator_finetune.py

echo "=================================================="
echo "🚀 終極優化計畫開始 (Phase 3/3)"
echo "=================================================="
echo "啟動 Stage 3: 純 L1 Loss 極限微調 (15 Epochs)..."
python finetune_flattop_cnn.py

echo "=================================================="
echo "✅ 所有終極訓練完成！完美大腦已儲存為 best_resnet_finetuned.pth"
echo "請至 Streamlit App 重新整理並測試 ideal_flattop.png"
echo "=================================================="
