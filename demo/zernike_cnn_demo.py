import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mplconfig')
import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, laplace, map_coordinates

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

# Fix random seeds for reproducibility.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Fifteen Zernike modes (including Piston). Piston is index 0 and will not be trained.
ZERNIKE_MODES: List[Tuple[int, int, str]] = [
    (0, 0, 'piston'),
    (1, -1, 'tilt_y'),
    (1, 1, 'tilt_x'),
    (2, -2, 'astig_45'),
    (2, 0, 'defocus'),
    (2, 2, 'astig_0'),
    (3, -3, 'trefoil_y'),
    (3, -1, 'coma_y'),
    (3, 1, 'coma_x'),
    (3, 3, 'trefoil_x'),
    (4, -4, 'quadrafoil_y'),
    (4, -2, 'secondary_astig_45'),
    (4, 0, 'primary_spherical'),
    (4, 2, 'secondary_astig_0'),
    (4, 4, 'quadrafoil_x'),
]

def make_grid(n: int = 64):
    x = np.linspace(-1.0, 1.0, n)
    y = np.linspace(-1.0, 1.0, n)
    xx, yy = np.meshgrid(x, y)
    rr = np.sqrt(xx**2 + yy**2)
    th = np.arctan2(yy, xx)
    pupil = rr <= 1.0
    return xx, yy, rr, th, pupil

def zernike_radial(n: int, m: int, r: np.ndarray) -> np.ndarray:
    m = abs(m)
    if (n - m) % 2 != 0:
        return np.zeros_like(r)
    out = np.zeros_like(r)
    for k in range((n - m) // 2 + 1):
        coeff = ((-1) ** k) * math.factorial(n - k)
        denom = (
            math.factorial(k)
            * math.factorial((n + m) // 2 - k)
            * math.factorial((n - m) // 2 - k)
        )
        out += coeff / denom * r ** (n - 2 * k)
    return out

def zernike(n: int, m: int, r: np.ndarray, theta: np.ndarray) -> np.ndarray:
    rad = zernike_radial(n, m, r)
    if m > 0:
        z = rad * np.cos(m * theta)
    elif m < 0:
        z = rad * np.sin(abs(m) * theta)
    else:
        z = rad
    mask = r <= 1.0
    z = z * mask
    std = z[mask].std()
    if std > 1e-8:
        z = z / std
    return z.astype(np.float32)

@dataclass
class ForwardModelConfig:
    image_size: int = 64
    coeff_std: float = 0.18
    warp_scale: float = 0.055
    blur_scale: float = 1.20
    shading_scale: float = 0.10
    noise_std: float = 0.005
    target_order: int = 10
    target_wx: float = 0.43
    target_wy: float = 0.25

class FlatTopZernikeSimulator:
    def __init__(self, cfg: ForwardModelConfig):
        self.cfg = cfg
        self.xx, self.yy, self.rr, self.th, self.pupil = make_grid(cfg.image_size)
        self.target = self._make_target()
        self.basis = self._build_basis()
        self.basis_dx = np.stack([np.gradient(b, axis=1) for b in self.basis], axis=0)
        self.basis_dy = np.stack([np.gradient(b, axis=0) for b in self.basis], axis=0)

    def _make_target(self):
        wx, wy, p = self.cfg.target_wx, self.cfg.target_wy, self.cfg.target_order
        tgt = np.exp(-((np.abs(self.xx) / wx) ** p + (np.abs(self.yy) / wy) ** p))
        return (tgt / tgt.max()).astype(np.float32)

    def _build_basis(self):
        return np.stack([zernike(n, m, self.rr, self.th) for n, m, _ in ZERNIKE_MODES], axis=0)

    def simulate(self, coeffs: np.ndarray, add_noise: bool = True):
        coeffs = np.asarray(coeffs, dtype=np.float32)
        wavefront = np.tensordot(coeffs, self.basis, axes=(0, 0))
        dx = np.tensordot(coeffs, self.basis_dx, axes=(0, 0))
        dy = np.tensordot(coeffs, self.basis_dy, axes=(0, 0))

        xw = self.xx + self.cfg.warp_scale * dx
        yw = self.yy + self.cfg.warp_scale * dy
        px = (xw + 1.0) * 0.5 * (self.cfg.image_size - 1)
        py = (yw + 1.0) * 0.5 * (self.cfg.image_size - 1)
        warped = map_coordinates(self.target, [py, px], order=1, mode='constant', cval=0.0)

        blur_sigma = self.cfg.blur_scale * float(np.sqrt(np.mean(coeffs**2)))
        if blur_sigma > 1e-6:
            warped = gaussian_filter(warped, sigma=blur_sigma)

        shading = 1.0 + self.cfg.shading_scale * laplace(wavefront)
        shading = np.clip(shading, 0.75, 1.25)
        image = np.clip(warped * shading, 0.0, None)
        image = image / (image.max() + 1e-8)

        if add_noise:
            image = image + np.random.normal(0.0, self.cfg.noise_std, size=image.shape).astype(np.float32)
        image = np.clip(image, 0.0, 1.0)
        return image.astype(np.float32)

    def flatness_metric(self, image: np.ndarray, roi_level: float = 0.8) -> float:
        mask = self.target > roi_level
        region = image[mask]
        return float(region.std() / (region.mean() + 1e-8))

class ZernikeDataset(Dataset):
    def __init__(self, simulator: FlatTopZernikeSimulator, num_samples: int = 1000):
        self.sim = simulator
        self.num_modes = len(ZERNIKE_MODES)
        self.coeffs = np.random.normal(0.0, self.sim.cfg.coeff_std, size=(num_samples, self.num_modes)).astype(np.float32)
        self.coeffs[:, 0] = 0.0 # Force Piston to be 0
        self.images = np.stack([self.sim.simulate(c, add_noise=True) for c in self.coeffs], axis=0)[:, None, :, :]

    def __len__(self):
        return len(self.coeffs)

    def __getitem__(self, idx):
        return torch.from_numpy(self.images[idx]), torch.from_numpy(self.coeffs[idx])

class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class ZernikeRegressor(nn.Module):
    def __init__(self, num_predict_modes: int):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 16),
            nn.MaxPool2d(2),
            ConvBlock(16, 32),
            nn.MaxPool2d(2),
            ConvBlock(32, 64),
            nn.MaxPool2d(2),
            ConvBlock(64, 128),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(128, num_predict_modes),
        )

    def forward(self, x):
        return self.head(self.features(x))

def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_mae = 0.0
    total_n = 0

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        y_train = y[:, 1:] # Skip Piston (index 0)
        pred = model(x)
        loss = criterion(pred, y_train)
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        mae = (pred - y_train).abs().mean().item()
        bs = x.size(0)
        total_loss += loss.item() * bs
        total_mae += mae * bs
        total_n += bs
    return total_loss / total_n, total_mae / total_n

def train_model(model, train_loader, val_loader, epochs=5, lr=1e-3):
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': []}
    best_state = None
    best_val = float('inf')

    for ep in range(1, epochs + 1):
        train_loss, train_mae = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_mae = run_epoch(model, val_loader, criterion, optimizer=None)
        scheduler.step()
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)
        print(f'Epoch {ep:02d}/{epochs} | train loss={train_loss:.5f}, val loss={val_loss:.5f}, train mae={train_mae:.5f}, val mae={val_mae:.5f}')
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return history

def predict_coeffs(model: nn.Module, image: np.ndarray):
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(image[None, None, :, :]).float().to(DEVICE)
        pred_14 = model(x).cpu().numpy()[0]
    
    # Reconstruct 15D vector with Piston=0
    pred_15 = np.zeros(len(ZERNIKE_MODES), dtype=np.float32)
    pred_15[1:] = pred_14
    return pred_15

def run_adaptive_loop(sim: FlatTopZernikeSimulator, model: nn.Module, true_coeffs: np.ndarray, steps: int = 8, gain: float = 0.85):
    correction = np.zeros_like(true_coeffs, dtype=np.float32)
    images = []
    residual_norms = []
    flatness = []
    predicted = []

    for _ in range(steps):
        residual = true_coeffs + correction
        image = sim.simulate(residual, add_noise=False)
        pred = predict_coeffs(model, image)
        correction = correction - gain * pred
        images.append(image)
        residual_norms.append(float(np.sqrt(np.mean(residual**2))))
        flatness.append(sim.flatness_metric(image))
        predicted.append(pred)

    final_residual = true_coeffs + correction
    final_image = sim.simulate(final_residual, add_noise=False)
    images.append(final_image)
    residual_norms.append(float(np.sqrt(np.mean(final_residual**2))))
    flatness.append(sim.flatness_metric(final_image))

    return {
        'images': images,
        'residual_norms': residual_norms,
        'flatness': flatness,
        'predicted': predicted,
        'final_residual': final_residual,
        'final_correction': correction,
    }

def plot_training(history, outdir):
    plt.figure(figsize=(7, 4))
    plt.plot(history['train_loss'], label='train loss')
    plt.plot(history['val_loss'], label='val loss')
    plt.xlabel('epoch')
    plt.ylabel('SmoothL1 loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'training_curve.png'), dpi=180)
    plt.close()

def plot_prediction_example(sim, model, dataset, outdir):
    idx = np.random.randint(len(dataset))
    image = dataset.images[idx, 0]
    true_coeffs = dataset.coeffs[idx]
    pred_coeffs = predict_coeffs(model, image)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(sim.target, cmap='inferno')
    axes[0].set_title('ideal flat-top target')
    axes[1].imshow(image, cmap='inferno')
    axes[1].set_title('input distorted image')
    axes[2].bar(np.arange(len(true_coeffs)), np.abs(true_coeffs - pred_coeffs))
    axes[2].set_title('|prediction error| by mode')
    axes[2].set_xlabel('mode index')
    for ax in axes[:2]:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'prediction_example.png'), dpi=180)
    plt.close()

def plot_adaptive_result(sim, result, outdir):
    initial = result['images'][0]
    final = result['images'][-1]
    steps = np.arange(len(result['residual_norms']))

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes[0, 0].imshow(sim.target, cmap='inferno')
    axes[0, 0].set_title('ideal target')
    axes[0, 1].imshow(initial, cmap='inferno')
    axes[0, 1].set_title(f"before correction\nflatness={result['flatness'][0]*100:.2f}%")
    axes[0, 2].imshow(final, cmap='inferno')
    axes[0, 2].set_title(f"after correction\nflatness={result['flatness'][-1]*100:.2f}%")
    axes[1, 0].plot(steps, result['residual_norms'], marker='o')
    axes[1, 0].set_title('residual coefficient RMS')
    axes[1, 0].set_xlabel('iteration')
    axes[1, 1].plot(steps, np.array(result['flatness']) * 100.0, marker='o')
    axes[1, 1].set_title('flatness non-uniformity (%)')
    axes[1, 1].set_xlabel('iteration')
    axes[1, 2].bar(np.arange(len(result['final_residual'])), result['final_residual'])
    axes[1, 2].set_title('final residual coefficients')
    axes[1, 2].set_xlabel('mode index')
    for ax in axes[0]:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'adaptive_correction.png'), dpi=180)
    plt.close()

def main():
    outdir = '../outputs/demo'
    os.makedirs(outdir, exist_ok=True)

    cfg = ForwardModelConfig()
    sim = FlatTopZernikeSimulator(cfg)
    dataset = ZernikeDataset(sim, num_samples=800)

    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(SEED))
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=0)

    model = ZernikeRegressor(num_predict_modes=len(ZERNIKE_MODES) - 1).to(DEVICE)
    print(f'Training on device: {DEVICE}')
    history = train_model(model, train_loader, val_loader, epochs=5, lr=1e-3)

    torch.save({
        'model_state': model.state_dict(),
        'mode_names': [name for _, _, name in ZERNIKE_MODES],
        'config': cfg.__dict__,
    }, os.path.join(outdir, 'zernike_cnn_model.pt'))

    plot_training(history, outdir)
    plot_prediction_example(sim, model, dataset, outdir)

    true_coeffs = np.random.normal(0.0, cfg.coeff_std, size=len(ZERNIKE_MODES)).astype(np.float32)
    true_coeffs[0] = 0.0 # Set Piston to 0
    result = run_adaptive_loop(sim, model, true_coeffs, steps=8, gain=0.85)
    plot_adaptive_result(sim, result, outdir)

    print('\nMode list:')
    for i, (_, _, name) in enumerate(ZERNIKE_MODES):
        print(f'{i:2d}: {name}')
    print('\nTrue coefficients:')
    print(np.round(true_coeffs, 4))
    print('\nFinal residual coefficients:')
    print(np.round(result['final_residual'], 4))
    print(f"\nInitial flatness non-uniformity: {result['flatness'][0]*100:.2f}%")
    print(f"Final flatness non-uniformity: {result['flatness'][-1]*100:.2f}%")
    print(f'\nOutputs saved to: {outdir}')

if __name__ == '__main__':
    main()
