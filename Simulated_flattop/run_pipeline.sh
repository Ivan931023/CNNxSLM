#!/bin/bash
source ../.venv/bin/activate
echo "Starting Dataset Generation (50,000 samples)..."
python dataset_generator.py

echo "Starting CNN Training on generated dataset..."
python train_flattop_cnn.py

echo "Pipeline Finished!"
