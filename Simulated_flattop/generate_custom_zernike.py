import torch
import numpy as np
import h5py
import os
import math
import matplotlib.pyplot as plt
import argparse

def IDFT(u):
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(u, dim=(-2, -1))), dim=(-2, -1))

def main():
    parser = argparse.ArgumentParser(description="Generate a flattop beam image with custom Zernike coefficients.")
    for i in range(1, 16):
        parser.add_argument(f'--Z{i}', type=float, default=0.0, help=f'Zernike mode {i} coefficient (default: 0.0)')
    parser.add_argument('--output', type=str, default='custom_flattop.png', help='Output filename (default: custom_flattop.png)')
    args = parser.parse_args()

    print("Loading tensors.mat...")
    if not os.path.exists('tensors.mat'):
        print("Error: tensors.mat not found! Please make sure it is in the current directory.")
        return

    mat = h5py.File('tensors.mat', 'r')
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    Input_beam = torch.tensor(mat['Input_beam'][()].T, dtype=torch.float32, device=device)
    Blazed_phi = torch.tensor(mat['Blazed_phi'][()].T, dtype=torch.float32, device=device)
    theta_rad = torch.tensor(mat['theta_rad'][()].T, dtype=torch.float32, device=device)
    Pupil_mask = torch.tensor(mat['Pupil_mask'][()].T, dtype=torch.float32, device=device)

    Z_basis_np = mat['Z_basis'][()].transpose(2, 1, 0)
    Z_basis = torch.tensor(Z_basis_np, dtype=torch.float32, device=device)

    v = int(mat['v'][()][0,0])
    h = int(mat['h'][()][0,0])
    crop_range = 250

    no_modulate_part = IDFT(Input_beam.to(torch.complex64))
    origin_power = torch.sum(torch.abs(no_modulate_part)**2)

    ratio = 0.99999
    no_modulate = 0.01
    power_phase = origin_power * (1 - no_modulate) * ratio
    power_amplitude = origin_power * (1 - no_modulate) * (1 - ratio)
    power_no_modulate = origin_power * no_modulate

    no_modulate_part = no_modulate_part * torch.sqrt(power_no_modulate / torch.sum(torch.abs(no_modulate_part)**2))

    # --- Setup Custom Zernike Modes ---
    Z_batch = torch.zeros(1, 15, device=device, dtype=torch.float32)
    # Z1 is Piston (index 0), Z2 is index 1, ..., Z15 is index 14
    for i in range(1, 16):
        val = getattr(args, f'Z{i}')
        Z_batch[0, i-1] = val
        if val != 0.0:
            print(f"Setting Z{i} = {val}")

    Z_sum = torch.einsum('bi,ixy->bxy', Z_batch, Z_basis)
    Angle_exp_z = (Z_sum + 1.0) * math.pi
    Angle_exp_z = Angle_exp_z * Pupil_mask

    Total_phase = Blazed_phi.unsqueeze(0) + theta_rad.unsqueeze(0) + Angle_exp_z
    phase = torch.fmod(Total_phase, 2 * math.pi)

    Input_batch = Input_beam.unsqueeze(0).to(torch.complex64)
    phase_complex = torch.exp(1j * phase.to(torch.complex64))

    E_phase = Input_batch * phase_complex
    E_amp = Input_batch * (phase.to(torch.complex64) / (2 * math.pi))

    phase_part = IDFT(E_phase)
    amp_part = IDFT(E_amp)

    phase_norm = torch.sqrt(power_phase / torch.sum(torch.abs(phase_part)**2, dim=(-2,-1), keepdim=True))
    phase_part = phase_part * phase_norm

    amp_norm = torch.sqrt(power_amplitude / torch.sum(torch.abs(amp_part)**2, dim=(-2,-1), keepdim=True))
    amp_part = amp_part * amp_norm

    result = phase_part + amp_part + no_modulate_part.unsqueeze(0)
    I = torch.abs(result)**2
    I_crop = I[:, v-crop_range:v+crop_range, h-crop_range:h+crop_range]

    img = I_crop[0].cpu().numpy()
    img_norm = (img / np.max(img) * 255).astype(np.uint8)

    plt.imsave(args.output, img_norm, cmap='turbo')
    print(f"\n✅ Successfully saved customized image to: {args.output}")
    print("="*50)
    print("Custom Zernike Coefficients Used:")
    for i in range(1, 16):
        val = getattr(args, f'Z{i}')
        if val != 0.0:
            print(f"  - Z{i:02d}: {val:.4f}")
    print("="*50)

if __name__ == '__main__':
    main()
