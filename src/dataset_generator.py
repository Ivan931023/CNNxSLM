import torch
import numpy as np
import h5py
import os
import math
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def IDFT(u):
    # u: [B, H, W] or [H, W]
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(u, dim=(-2, -1))), dim=(-2, -1))

def main():
    print("Loading tensors.mat...")
    if not os.path.exists('tensors.mat'):
        print("tensors.mat not found! Run export_static_tensors.m in MATLAB first.")
        return
        
    mat = h5py.File('tensors.mat', 'r')
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Extract arrays and move to GPU
    # h5py reads MATLAB (H,W) as (W,H), so we transpose (.T)
    Input_beam = torch.tensor(mat['Input_beam'][()].T, dtype=torch.float32, device=device)
    Blazed_phi = torch.tensor(mat['Blazed_phi'][()].T, dtype=torch.float32, device=device)
    theta_rad = torch.tensor(mat['theta_rad'][()].T, dtype=torch.float32, device=device)
    Pupil_mask = torch.tensor(mat['Pupil_mask'][()].T, dtype=torch.float32, device=device)
    
    # h5py reads MATLAB (15,H,W) as (W,H,15), we permute to (15,H,W)
    Z_basis_np = mat['Z_basis'][()].transpose(2, 1, 0)
    Z_basis = torch.tensor(Z_basis_np, dtype=torch.float32, device=device)
    
    # Crop center coordinates
    v = int(mat['v'][()][0,0])
    h = int(mat['h'][()][0,0])
    crop_range = 250
    print(f"Crop center: (v={v}, h={h}), Crop range: {crop_range}")
    
    # Precompute constant parts
    print("Precomputing static IDFT components...")
    no_modulate_part = IDFT(Input_beam.to(torch.complex64))
    origin_power = torch.sum(torch.abs(no_modulate_part)**2)
    
    ratio = 0.99999
    no_modulate = 0.01
    power_phase = origin_power * (1 - no_modulate) * ratio
    power_amplitude = origin_power * (1 - no_modulate) * (1 - ratio)
    power_no_modulate = origin_power * no_modulate
    
    no_modulate_part = no_modulate_part * torch.sqrt(power_no_modulate / torch.sum(torch.abs(no_modulate_part)**2))
    
    # Dataset config
    num_samples = 50000  # Full dataset
    batch_size = 1

    
    os.makedirs('dataset', exist_ok=True)
    csv_file = open('dataset/labels.csv', 'w', newline='')
    writer = csv.writer(csv_file)
    headers = ['filename'] + [f'Z{i+1}' for i in range(15)]
    writer.writerow(headers)
    
    print(f"Generating {num_samples} samples in batches of {batch_size}...")
    
    sample_idx = 0
    for batch_start in tqdm(range(0, num_samples, batch_size)):
        bs = min(batch_size, num_samples - batch_start)
        
        # 1. Random Zernike coefficients [-2.0, 2.0]
        # Shape: [B, 15]
        Z_batch = torch.rand(bs, 15, device=device, dtype=torch.float32) * 4.0 - 2.0
        # Optional: set Piston to 0
        # Z_batch[:, 0] = 0
        
        # 2. Compute Zernike Phase
        # einsum: b=batch, i=zernike_mode, x,y=spatial
        Z_sum = torch.einsum('bi,ixy->bxy', Z_batch, Z_basis)
        Angle_exp_z = (Z_sum + 1.0) * math.pi
        Angle_exp_z = Angle_exp_z * Pupil_mask
        
        # 3. Total Phase
        Total_phase = Blazed_phi.unsqueeze(0) + theta_rad.unsqueeze(0) + Angle_exp_z
        phase = torch.fmod(Total_phase, 2 * math.pi)
        
        # 4. Create E-fields
        Input_batch = Input_beam.unsqueeze(0).to(torch.complex64)
        phase_complex = torch.exp(1j * phase.to(torch.complex64))
        
        E_phase = Input_batch * phase_complex
        E_amp = Input_batch * (phase.to(torch.complex64) / (2 * math.pi))
        
        # 5. IDFT
        phase_part = IDFT(E_phase)
        amp_part = IDFT(E_amp)
        
        # Normalization
        phase_norm = torch.sqrt(power_phase / torch.sum(torch.abs(phase_part)**2, dim=(-2,-1), keepdim=True))
        phase_part = phase_part * phase_norm
        
        amp_norm = torch.sqrt(power_amplitude / torch.sum(torch.abs(amp_part)**2, dim=(-2,-1), keepdim=True))
        amp_part = amp_part * amp_norm
        
        # 6. Combine
        result = phase_part + amp_part + no_modulate_part.unsqueeze(0)
        I = torch.abs(result)**2
        
        # 7. Crop
        I_crop = I[:, v-crop_range:v+crop_range, h-crop_range:h+crop_range]
        
        # Save to disk
        I_np = I_crop.cpu().numpy()
        Z_np = Z_batch.cpu().numpy()
        
        # Free memory
        del phase, Total_phase, E_phase, E_amp, phase_part, amp_part, result, I, I_crop, Input_batch, phase_complex
        if device.type == 'mps':
            torch.mps.empty_cache()
        elif device.type == 'cuda':
            torch.cuda.empty_cache()
        
        for b in range(bs):
            img = I_np[b]
            z_vals = Z_np[b]
            
            # Normalize image to 0-255
            img_norm = (img / np.max(img) * 255).astype(np.uint8)
            
            filename = f"sample_{sample_idx:05d}.png"
            plt.imsave(os.path.join('dataset', filename), img_norm, cmap='turbo')
            
            writer.writerow([filename] + z_vals.tolist())
            sample_idx += 1

    csv_file.close()
    print("Dataset generation completed!")

if __name__ == '__main__':
    main()
