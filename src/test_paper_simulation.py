import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.fft import fft2, fftshift, ifftshift

def blazed_grating_rotate(pixel, max_val, min_val, levels, repeat, theta):
    x = np.arange(1, pixel + 1)
    y = np.arange(1, pixel + 1)
    X, Y = np.meshgrid(x, y)
    
    R = np.array([[np.cos(theta), -np.sin(theta)], 
                  [np.sin(theta), np.cos(theta)]])
    
    coords = R @ np.vstack([X.ravel(), Y.ravel()])
    X_rot = coords[0, :].reshape(X.shape)
    
    grating = (np.floor(X_rot / repeat) % levels) * (max_val - min_val) / (levels - 1) + min_val
    return grating

def grating_phase(pixel, max_val, min_val, levels, repeat, theta_blazed):
    grat = blazed_grating_rotate(pixel, max_val, min_val, levels, repeat, theta_blazed)
    grat = 2 * np.pi / 255 * grat
    return grat

def gaussian_beam(beam_size, pixel, dx):
    Nx = int(pixel)
    Ny = int(pixel)
    dy = dx
    x = np.arange(-Nx/2, Nx/2) * dx
    y = np.arange(-Ny/2, Ny/2) * dy
    X, Y = np.meshgrid(x, -y)
    
    sig_x = beam_size / 4.0
    sig_y = beam_size / 4.0
    
    input_beam = np.exp(-(X**2 / (2 * sig_x**2) + Y**2 / (2 * sig_y**2)))
    input_beam = input_beam / np.max(input_beam)
    return input_beam

def padding(target, size):
    pixel = int(size)
    l, k = target.shape
    data = np.zeros((pixel, pixel))
    
    start_i = (pixel - l) // 2
    start_j = (pixel - k) // 2
    data[start_i:start_i+l, start_j:start_j+k] = target
    return data

def dft(u):
    return fftshift(fft2(ifftshift(u)))

def main():
    f = 300.0                 # Focal length (mm)
    lam = 447e-6              # Wave length (mm)
    CCD_pixel_size = 2.2e-3   # (mm)
    dx = 8e-3                 # (mm)
    
    # Calculate pixel size according to Fourier optics
    val = lam * f / (dx * CCD_pixel_size)
    pixel = int(np.round(val / 10.0) * 10)
    
    pixel_grating = 1000
    beam_size = 1.0
    range_val = 1000
    
    r = 0.055
    t = 1 - r
    w = 6e-3
    
    theta_deg = 0
    theta_blazed = np.deg2rad(theta_deg)
    max_phase = 255
    min_phase = 0
    repeat = 1
    level = 20
    
    print(f"Calculated Simulation Grid: {pixel}x{pixel}")
    
    print("Generating Grating Phase...")
    blazed_theta = grating_phase(pixel_grating, max_phase, min_phase, level, repeat, theta_blazed)
    
    print("Generating Gaussian Beam...")
    input_beam = gaussian_beam(beam_size, pixel, dx)
    
    print("Padding Phase...")
    phi = padding(blazed_theta, pixel)
    
    print("Calculating E field...")
    E = -(r + np.exp(1j * phi)) / (1 + r * np.exp(1j * phi))
    
    print("Executing DFT (this might take a few seconds)...")
    image = np.abs(dft(input_beam * E))**2
    
    c = pixel // 2
    data_zoomin = image[c - range_val : c + range_val + 1, c - range_val : c + range_val + 1]
    
    outdir = '../outputs/simulation'
    os.makedirs(outdir, exist_ok=True)
    
    print("Saving plots...")
    # 1. Phase pattern
    plt.figure()
    plt.imshow(np.mod(phi, 2*np.pi), vmin=0, vmax=2*np.pi, cmap='gray')
    plt.colorbar()
    plt.title("Phase Pattern (mod 2pi)")
    plt.savefig(os.path.join(outdir, 'phase_pattern.png'))
    plt.close()
    
    # 2. Full diffraction image
    plt.figure()
    plt.imshow(image, cmap='turbo')
    plt.colorbar()
    plt.title("Full Diffraction Image")
    plt.savefig(os.path.join(outdir, 'full_image.png'))
    plt.close()
    
    # 3. Zoomed-in image
    plt.figure()
    plt.imshow(data_zoomin, cmap='turbo', vmin=0, vmax=np.max(data_zoomin))
    plt.colorbar()
    plt.title("Zoomed-in Diffraction Image")
    plt.savefig(os.path.join(outdir, 'zoomin_image.png'))
    plt.close()
    
    # 4. Cross-section
    plt.figure()
    plt.plot(np.arange(data_zoomin.shape[1]), data_zoomin[range_val, :])
    plt.xlim([0, 2*range_val])
    plt.title("Cross-section Intensity")
    plt.savefig(os.path.join(outdir, 'cross_section.png'))
    plt.close()
    
    print("Simulation complete! Outputs saved in", outdir)

if __name__ == '__main__':
    main()
