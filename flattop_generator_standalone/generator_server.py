#!/usr/bin/env python3
"""Standalone HTTP server for the Flattop Beam Generator web UI.

All paths are relative to this script's directory, so the folder is fully
self-contained: copy the folder anywhere, install requirements, run.
"""

import io
import json
import math
import os
import sys
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

PORT = 8766
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAT_PATH = os.path.join(SCRIPT_DIR, 'tensors.mat')

_state = {}


def IDFT(u):
    return torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.ifftshift(u, dim=(-2, -1))),
        dim=(-2, -1),
    )


def init_simulation():
    if not os.path.exists(MAT_PATH):
        print(f"❌ tensors.mat not found at {MAT_PATH}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {MAT_PATH} (this takes a few seconds)...")
    mat = h5py.File(MAT_PATH, 'r')

    device = torch.device(
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )
    print(f"Using device: {device}")

    Input_beam = torch.tensor(mat['Input_beam'][()].T, dtype=torch.float32, device=device)
    Blazed_phi = torch.tensor(mat['Blazed_phi'][()].T, dtype=torch.float32, device=device)
    theta_rad = torch.tensor(mat['theta_rad'][()].T, dtype=torch.float32, device=device)
    Pupil_mask = torch.tensor(mat['Pupil_mask'][()].T, dtype=torch.float32, device=device)
    Z_basis = torch.tensor(
        mat['Z_basis'][()].transpose(2, 1, 0), dtype=torch.float32, device=device
    )
    v = int(mat['v'][()][0, 0])
    h = int(mat['h'][()][0, 0])
    crop_range = 250

    no_modulate_part = IDFT(Input_beam.to(torch.complex64))
    origin_power = torch.sum(torch.abs(no_modulate_part) ** 2)

    ratio = 0.99999
    no_modulate = 0.01
    power_phase = origin_power * (1 - no_modulate) * ratio
    power_amplitude = origin_power * (1 - no_modulate) * (1 - ratio)
    power_no_modulate = origin_power * no_modulate

    no_modulate_part = no_modulate_part * torch.sqrt(
        power_no_modulate / torch.sum(torch.abs(no_modulate_part) ** 2)
    )

    static_phase = Blazed_phi + theta_rad

    _state.update({
        'device': device,
        'Input_beam': Input_beam,
        'static_phase': static_phase,
        'Pupil_mask': Pupil_mask,
        'Z_basis': Z_basis,
        'v': v, 'h': h, 'crop_range': crop_range,
        'no_modulate_part': no_modulate_part,
        'power_phase': power_phase,
        'power_amplitude': power_amplitude,
    })
    print("✅ Simulation tensors loaded and ready.")


@torch.no_grad()
def generate_image(z2_to_z15):
    s = _state
    device = s['device']

    z_full = [0.0] + [float(v) for v in z2_to_z15]
    Z_batch = torch.tensor([z_full], dtype=torch.float32, device=device)

    Z_sum = torch.einsum('bi,ixy->bxy', Z_batch, s['Z_basis'])
    Angle_exp_z = (Z_sum + 1.0) * math.pi
    Angle_exp_z = Angle_exp_z * s['Pupil_mask']

    Total_phase = s['static_phase'].unsqueeze(0) + Angle_exp_z
    phase = torch.fmod(Total_phase, 2 * math.pi)

    Input_batch = s['Input_beam'].unsqueeze(0).to(torch.complex64)
    phase_complex = torch.exp(1j * phase.to(torch.complex64))

    E_phase = Input_batch * phase_complex
    E_amp = Input_batch * (phase.to(torch.complex64) / (2 * math.pi))

    phase_part = IDFT(E_phase)
    amp_part = IDFT(E_amp)

    phase_norm = torch.sqrt(
        s['power_phase'] / torch.sum(torch.abs(phase_part) ** 2, dim=(-2, -1), keepdim=True)
    )
    phase_part = phase_part * phase_norm

    amp_norm = torch.sqrt(
        s['power_amplitude'] / torch.sum(torch.abs(amp_part) ** 2, dim=(-2, -1), keepdim=True)
    )
    amp_part = amp_part * amp_norm

    result = phase_part + amp_part + s['no_modulate_part'].unsqueeze(0)
    I = torch.abs(result) ** 2

    v, h, cr = s['v'], s['h'], s['crop_range']
    I_crop = I[:, v - cr:v + cr, h - cr:h + cr]

    img = I_crop[0].cpu().numpy()
    img_norm = (img / np.max(img) * 255).astype(np.uint8)

    buf = io.BytesIO()
    plt.imsave(buf, img_norm, cmap='turbo', format='png')
    return buf.getvalue()


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=SCRIPT_DIR, **kwargs)

    def log_message(self, fmt, *args):
        sys.stderr.write(f"[{self.log_date_time_string()}] {fmt % args}\n")

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        if urlparse(self.path).path != '/api/generate':
            self.send_response(404)
            self.end_headers()
            return

        try:
            length = int(self.headers.get('content-length', 0))
            payload = json.loads(self.rfile.read(length).decode('utf-8'))
            coeffs = [float(payload.get(f'Z{i}', 0.0)) for i in range(2, 16)]
            png = generate_image(coeffs)

            self.send_response(200)
            self.send_header('Content-Type', 'image/png')
            self.send_header('Content-Length', str(len(png)))
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(png)
        except Exception as e:
            err = f"Generation failed: {type(e).__name__}: {e}"
            print(err, file=sys.stderr)
            self.send_response(500)
            self.send_header('Content-Type', 'text/plain; charset=utf-8')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(err.encode('utf-8'))


def main():
    init_simulation()
    server = ThreadingHTTPServer(('127.0.0.1', PORT), Handler)
    url = f"http://localhost:{PORT}/Flattop_beam_with_Zernike_aberrations.html"
    print()
    print(f"🚀 Generator server running at:\n   {url}")
    print("\nPress Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == '__main__':
    main()
