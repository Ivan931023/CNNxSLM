# CNNxSLM — CNN-Based Zernike Aberration Correction for SLM Flattop Beams

This project uses a Convolutional Neural Network (CNN) to predict Zernike polynomial coefficients from distorted flattop beam intensity images, enabling real-time adaptive aberration correction for Spatial Light Modulators (SLMs).

## 📁 Project Structure

```
CNNxSLM/
├── src/                  # Core Python source code
│   ├── dataset_generator.py          # Generate training data from MATLAB tensors
│   ├── dataset_generator_finetune.py # Generate finetune data
│   ├── train_flattop_cnn.py          # Stage 1: CNN training (ResNet-34)
│   ├── finetune_flattop_cnn.py       # Stage 2: Finetune with L1 sparsity
│   ├── generate_custom_zernike.py    # Generate beam with custom Zernike coeffs
│   ├── generate_ideal.py             # Generate ideal (zero-aberration) beam
│   └── test_paper_simulation.py      # Reproduce paper simulation
│
├── app/                  # Streamlit web application
│   ├── app.py            # v1: ResNet-18, 128×128
│   └── app_v2.py         # v2: ResNet-34, 500×500 (Hyper-Precision)
│
├── matlab/               # Original MATLAB simulation code
├── scripts/              # Shell scripts for training pipelines
├── demo/                 # Early-stage demo code and outputs
├── docs/                 # Reports, slides, and documentation
├── data/                 # Datasets, tensors.mat (gitignored)
├── outputs/              # Generated images, models, simulation results
└── logs/                 # Training logs (gitignored)
```

## 🚀 Quick Start

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Generate dataset (requires data/tensors.mat from MATLAB)
cd src && python dataset_generator.py

# 3. Train the model
python train_flattop_cnn.py

# 4. Finetune
python finetune_flattop_cnn.py

# 5. Launch the web app
cd ../app && streamlit run app_v2.py
```

## 🔬 Key Features

- **ResNet-34 backbone** modified for single-channel grayscale input
- **15 Zernike modes** (Z1–Z15) prediction, excluding piston
- **Two-stage training**: Stage 1 (base) → Stage 2 (L1 sparsity finetune)
- **Streamlit web app** for real-time beam analysis
- **MATLAB ↔ Python pipeline** using exported tensors

## 📦 Requirements

```bash
pip install -r requirements.txt
```

## 📄 License

For academic use. Please cite the associated paper if used in publications.
