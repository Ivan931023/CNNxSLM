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

## 🌐 Browser Apps (no Streamlit required)

Two self-contained HTML pages drive the trained model and the simulator from
the browser:

| Page | What it does | Backend |
| --- | --- | --- |
| [`Zernike_predictor.html`](Zernike_predictor.html) | Drag in a flattop image, get Z2–Z15 predicted in-browser via ONNX Runtime Web | Static — needs only `outputs/models/zernike_resnet_single.onnx` and a local server |
| [`Flattop_beam_with_Zernike_aberrations.html`](Flattop_beam_with_Zernike_aberrations.html) | Set Z2–Z15 with sliders, generate the resulting flattop beam, download the PNG | `app/generator_server.py` (loads `data/tensors.mat`) |

Launchers (double-click on macOS):

- `launch.command` — predictor (port 8765)
- `launch_generator.command` / `.sh` / `.bat` — generator (port 8766, cross-platform)

A self-contained copy of the generator lives in
[`flattop_generator_standalone/`](flattop_generator_standalone/) — copy that
folder anywhere, drop `tensors.mat` into it (download from Releases), install
requirements, then double-click `launch.command` (or the `.sh` / `.bat` for
Linux / Windows). See its own `README.md` for step-by-step instructions.

## 📥 Model & Data Files (not in repo)

These large binaries are **gitignored** and must be downloaded separately
from the [**v1.0-models release**](https://github.com/Ivan931023/CNNxSLM/releases/tag/v1.0-models):

| File | Size | Place in | Used by |
| --- | --- | --- | --- |
| [`best_resnet_finetuned.pth`](https://github.com/Ivan931023/CNNxSLM/releases/download/v1.0-models/best_resnet_finetuned.pth) | ~81 MB | `outputs/models/` | `app/app_v2.py`, `scripts/export_onnx.py` |
| [`zernike_resnet_single.onnx`](https://github.com/Ivan931023/CNNxSLM/releases/download/v1.0-models/zernike_resnet_single.onnx) | ~81 MB | `outputs/models/` | `Zernike_predictor.html` |
| [`tensors.mat`](https://github.com/Ivan931023/CNNxSLM/releases/download/v1.0-models/tensors.mat) | ~65 MB | `data/` *and* `flattop_generator_standalone/` | All `src/*.py` simulators, `app/generator_server.py`, and the standalone folder |

Quick download (macOS / Linux):

```bash
mkdir -p outputs/models data
BASE=https://github.com/Ivan931023/CNNxSLM/releases/download/v1.0-models
curl -L -o outputs/models/best_resnet_finetuned.pth   $BASE/best_resnet_finetuned.pth
curl -L -o outputs/models/zernike_resnet_single.onnx  $BASE/zernike_resnet_single.onnx
curl -L -o data/tensors.mat                           $BASE/tensors.mat
cp data/tensors.mat flattop_generator_standalone/tensors.mat
```

If you'd rather rebuild from scratch:

- Train models: `python src/train_flattop_cnn.py` then `src/finetune_flattop_cnn.py`
- Re-export ONNX: `python scripts/export_onnx.py`
- Regenerate datasets: `python src/dataset_generator.py` (needs `tensors.mat`)

## 📄 License

For academic use. Please cite the associated paper if used in publications.
