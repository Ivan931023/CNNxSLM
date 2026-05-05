# Flattop Beam Generator — Standalone

Self-contained version of the SLM Flattop Beam Generator. Everything required
to run `Flattop_beam_with_Zernike_aberrations.html` lives in this folder —
copy it anywhere, install the dependencies, and run.

## What's inside

| File | Purpose |
| --- | --- |
| `Flattop_beam_with_Zernike_aberrations.html` | Web UI (14 sliders for Z2–Z15, generate / download buttons) |
| `generator_server.py` | Python backend (HTTP server + SLM physics simulation) |
| `requirements.txt` | Python dependencies |
| `launch.command` | macOS launcher (double-click in Finder) |
| `launch.sh` | Linux launcher (`bash launch.sh`) |
| `launch.bat` | Windows launcher (double-click) |
| `tensors.mat` | **Not in this folder yet — see step 1 below.** Pre-computed pupil masks, Zernike basis, and Blazed phase (~65 MB) |

## Setup (one-time)

You need **Python 3.8+** installed.

### 1. Download `tensors.mat`

The 65 MB simulation tensor file is distributed via **GitHub Releases**:

> **GitHub → Releases →** download `tensors.mat` and drop it into **this
> folder** (alongside `generator_server.py`).

Without `tensors.mat`, `generator_server.py` will exit immediately with a
`tensors.mat not found` error.

### 2. Install Python dependencies

```bash
# (Optional but recommended) create a virtual environment
python3 -m venv .venv
source .venv/bin/activate         # macOS / Linux
# .venv\Scripts\activate          # Windows

# Install dependencies (~2 GB if PyTorch wheels not cached)
pip install -r requirements.txt
```

The launcher will automatically use `.venv/` if it exists; otherwise it falls
back to the system `python3` / `python`.

## Run

| Platform | Action |
| --- | --- |
| macOS | Double-click **`launch.command`** in Finder |
| Linux | `bash launch.sh` (or `chmod +x launch.sh && ./launch.sh`) |
| Windows | Double-click **`launch.bat`** |

The launcher:
1. Starts `generator_server.py` on `http://localhost:8766`
2. Loads `tensors.mat` into memory (takes a few seconds on GPU, longer on CPU)
3. Opens the HTML page in your default browser

Adjust Z2–Z15 with the sliders or numeric inputs, click **Generate Image**,
then **Download PNG** to save the result.

## Performance expectations

| Device | Single image |
| --- | --- |
| NVIDIA GPU (CUDA) | ~1–3 s |
| Apple Silicon (MPS) | ~4–7 s |
| CPU only | ~15–60 s |

Tensors loaded into RAM total ~4 GB, so the machine should have **at least
8 GB RAM free** during the run.

## Troubleshooting

- **`tensors.mat not found`** — the file must sit next to `generator_server.py`
  in this folder.
- **`Server offline` shown in the UI** — the launcher window probably crashed.
  Open a terminal in this folder and run `python3 generator_server.py` to see
  the error.
- **Port 8766 already in use** — edit `PORT` in both `generator_server.py` and
  the launcher you're using.
- **Browser shows a blank page** — open `http://localhost:8766/Flattop_beam_with_Zernike_aberrations.html`
  manually.

## How it works

The server simulates a Spatial Light Modulator (SLM) at the Fraunhofer plane:

1. Build a phase pattern from your Zernike coefficients × the basis in `tensors.mat`
2. Add the constant blazed-grating phase and tilt
3. Modulate the input beam by `exp(i·φ)` and propagate via inverse FFT
4. Add the small unmodulated and amplitude components
5. Crop the central 500×500 region and colorize with the `turbo` colormap

The numerical pipeline is identical to `src/generate_custom_zernike.py` from
the parent project — generated images are byte-for-byte the same.
