#!/bin/bash
set -e

echo "Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Running Zernike CNN Demo..."
python zernike_cnn_demo.py

echo "Done! Check the demo_outputs directory for results."
