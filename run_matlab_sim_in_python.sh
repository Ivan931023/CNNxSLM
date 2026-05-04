#!/bin/bash
set -e

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Running MATLAB Translation Simulation..."
python test_paper_simulation.py
