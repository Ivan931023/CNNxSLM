"""
SLM-CCD beam-centering + beam-size simulation.

Pipeline:
    Stage 1 (coarse centre): 1D knife-edge sweeps along x and y.
        Differentiate the transition curve to recover the 1D projection of
        |E|^2 and take its first moment (centroid). Robust for any beam shape;
        replaces the original code's "50%-point of the raw curve" approach,
        which was fragile (and depended on interp1d with non-sorted input).

    Stage 2 (fine centre): 2D test-spot scan around the coarse estimate. The
        intensity map is the beam profile convolved with a disk of radius
        R_spot. We fit a 2D Gaussian (LM) to recover the centre to ~0.05 px,
        falling back to a weighted centroid if the fit fails. The original
        code used argmax → resolution capped at the scan step.

    Stage 3 (beam size): with the centre found, sweep the radius of a
        concentric circular grating mask. The +1-order intensity follows
            I(R) = A · ( 1 - exp(-2 R^2 / w^2) ),
        from which the 1/e^2 waist w is recovered by a non-linear fit
        (with linear-regression seeding). Beam size is reported as the
        1/e^2 intensity radius.

Physics shortcut:
    For a binary 0/π Ronchi grating gated by a 0/1 mask M(x,y), the +1
    diffraction-order intensity equals |c_+1|^2 * integral( |E|^2 * M ).
    Using this avoids the full 1920x1080 FFT per scan step. The full-FFT
    version is still available via `capture_ccd_snapshot` for diagnostics.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.optimize import curve_fit


# --- 1. Environment --------------------------------------------------------
N, M = 1080, 1920                   # SLM  (height N, width M, landscape)
TRUE_CENTER = (1100.0, 540.0)       # (x, y) in pixels — entirely inside SLM
TRUE_W0 = 375.0                     # 1/e^2 intensity radius -> diameter 750 px
GRATING_PERIOD = 4
SHIFT_Y = N // GRATING_PERIOD
ORDER_1_POS = (N // 2 + SHIFT_Y, M // 2)

y_grid, x_grid = np.mgrid[0:N, 0:M]


def gaussian_beam(center, w0=TRUE_W0):
    cx, cy = center
    # Field amplitude: exp(-r^2 / w0^2)  =>  intensity = exp(-2 r^2 / w0^2)
    return np.exp(-((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / w0 ** 2)


# --- 2. Optical model ------------------------------------------------------
def apply_grating_mask(mask):
    grating = (np.pi * (y_grid % GRATING_PERIOD < GRATING_PERIOD / 2)).astype(float)
    return grating * mask


def capture_ccd_snapshot(E_in, phase):
    """Full 2f-system FFT — kept for diagnostic visualization only."""
    field = E_in * np.exp(1j * phase)
    return np.abs(np.fft.fftshift(np.fft.fft2(field))) ** 2


def first_order_intensity_fast(intensity_map, mask):
    """Closed-form: I_+1 ∝ ∫ |E|^2 · M dA."""
    return float((intensity_map * mask).sum())


# --- 3. Algorithm building blocks -----------------------------------------
def _centroid_from_knifeedge(steps, vals):
    steps = np.asarray(steps, dtype=float)
    vals = np.asarray(vals, dtype=float)
    proj = -np.diff(vals)
    if proj.sum() <= 0:
        proj = np.diff(vals)
    bin_centers = 0.5 * (steps[1:] + steps[:-1])
    proj = np.clip(proj, 0, None)
    if proj.sum() == 0:
        return float(np.mean(steps)), float(steps[-1] - steps[0])
    centroid = float(np.sum(bin_centers * proj) / proj.sum())
    pmax = proj.max(); half = pmax / 2.0
    # linear interpolation between adjacent bins straddling the half-max level
    lo = hi = float('nan')
    for i in range(len(proj) - 1):
        if proj[i] < half <= proj[i + 1]:
            t = (half - proj[i]) / (proj[i + 1] - proj[i])
            lo = bin_centers[i] + t * (bin_centers[i + 1] - bin_centers[i])
            break
    for i in range(len(proj) - 1, 0, -1):
        if proj[i] < half <= proj[i - 1]:
            t = (half - proj[i]) / (proj[i - 1] - proj[i])
            hi = bin_centers[i] + t * (bin_centers[i - 1] - bin_centers[i])
            break
    fwhm = float(hi - lo) if (np.isfinite(lo) and np.isfinite(hi) and hi > lo) else float(steps[1] - steps[0])
    return centroid, fwhm


def _gauss2d(xy, A, x0, y0, sx, sy, off):
    x, y = xy
    return (A * np.exp(-((x - x0) ** 2 / (2 * sx ** 2) + (y - y0) ** 2 / (2 * sy ** 2))) + off).ravel()


def _subpixel_fit(fine_results, offsets):
    yy, xx = np.meshgrid(offsets, offsets, indexing='ij')
    z = fine_results
    z0 = z - z.min()
    if z0.sum() == 0:
        return 0.0, 0.0
    x0 = float(np.sum(xx * z0) / z0.sum())
    y0 = float(np.sum(yy * z0) / z0.sum())
    try:
        span = max(2.0, (offsets[-1] - offsets[0]) / 6)
        p0 = (z.max() - z.min(), x0, y0, span, span, z.min())
        popt, _ = curve_fit(_gauss2d, (xx, yy), z.ravel(), p0=p0, maxfev=2000)
        return float(popt[1]), float(popt[2])
    except Exception:
        return x0, y0


# --- 4. Main pipeline ------------------------------------------------------
@dataclass
class CalibrationResult:
    coarse: tuple[float, float]
    fine: tuple[float, float]
    w0: float
    fine_results: np.ndarray
    fine_offsets: np.ndarray
    coarse_x_curve: tuple[np.ndarray, np.ndarray]
    coarse_y_curve: tuple[np.ndarray, np.ndarray]
    radii: np.ndarray
    radius_intensity: np.ndarray
    fwhm_x: float
    fwhm_y: float


def calibrate(E_in, *, coarse_step=50, fine_step=2, r_spot=50,
              radius_step=20, radius_max=None,
              first_order_fn=first_order_intensity_fast):
    intensity_map = np.abs(E_in) ** 2

    # ---- Stage 1: coarse knife-edge ----
    x_steps = np.arange(0, M, coarse_step)
    p_x = np.array([first_order_fn(intensity_map, x_grid > sx) for sx in x_steps])
    y_steps = np.arange(0, N, coarse_step)
    p_y = np.array([first_order_fn(intensity_map, y_grid > sy) for sy in y_steps])
    coarse_x, fwhm_x = _centroid_from_knifeedge(x_steps, p_x)
    coarse_y, fwhm_y = _centroid_from_knifeedge(y_steps, p_y)

    # ---- Stage 2: 2D fine scan, half-window from coarse FWHM (capped) ----
    half_window = min(40, max(15, int(0.08 * max(fwhm_x, fwhm_y))))
    offsets = np.arange(-half_window, half_window + 1, fine_step)
    fine_results = np.zeros((len(offsets), len(offsets)))
    for i, dy in enumerate(offsets):
        for j, dx in enumerate(offsets):
            tx, ty = coarse_x + dx, coarse_y + dy
            mask = ((x_grid - tx) ** 2 + (y_grid - ty) ** 2) <= r_spot ** 2
            fine_results[i, j] = first_order_fn(intensity_map, mask)
    dx_sub, dy_sub = _subpixel_fit(fine_results, offsets)
    fine_x = coarse_x + dx_sub
    fine_y = coarse_y + dy_sub

    # ---- Stage 3: radial scan -> 1/e^2 beam size ----
    if radius_max is None:
        # extend radius to ~2× expected waist  (use FWHM as proxy)
        radius_max = int(min(min(fine_x, M - fine_x, fine_y, N - fine_y),
                             1.6 * max(fwhm_x, fwhm_y) / 2.0))
    radii = np.arange(radius_step, radius_max + 1, radius_step)
    rr2 = (x_grid - fine_x) ** 2 + (y_grid - fine_y) ** 2
    radius_intensity = np.array([
        first_order_fn(intensity_map, rr2 <= R ** 2) for R in radii
    ])

    A_inf = radius_intensity[-1] * 1.05  # rough cap
    def _model(R, A, w):
        return A * (1.0 - np.exp(-2.0 * R ** 2 / w ** 2))
    try:
        # linear seeding via ln(1 - I/A) = -2R²/w²  (use middle 30-95 % range)
        norm = radius_intensity / max(A_inf, 1e-30)
        m = (norm > 0.05) & (norm < 0.97)
        if m.sum() >= 3:
            slope = np.polyfit(radii[m] ** 2, np.log(np.clip(1 - norm[m], 1e-12, None)), 1)[0]
            w_seed = float(np.sqrt(-2.0 / slope)) if slope < 0 else 200.0
        else:
            w_seed = 200.0
        popt, _ = curve_fit(_model, radii, radius_intensity,
                            p0=(A_inf, w_seed), maxfev=4000)
        w_fit = float(abs(popt[1]))
    except Exception:
        w_fit = float('nan')

    return CalibrationResult(
        coarse=(coarse_x, coarse_y), fine=(fine_x, fine_y), w0=w_fit,
        fine_results=fine_results, fine_offsets=offsets,
        coarse_x_curve=(x_steps, p_x), coarse_y_curve=(y_steps, p_y),
        radii=radii, radius_intensity=radius_intensity,
        fwhm_x=fwhm_x, fwhm_y=fwhm_y,
    )


# --- 5. Demo plot ----------------------------------------------------------
def _plot(result, true_center, true_w0, E_in):
    cx, cy = true_center
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Coarse curves
    xs, px = result.coarse_x_curve
    ys, py = result.coarse_y_curve
    ax = axes[0, 0]
    ax.plot(xs, px / px.max(), 'b.-', label='X (knife-edge)')
    ax.plot(ys, py / py.max(), 'r.-', label='Y (knife-edge)')
    ax.axvline(result.coarse[0], color='b', ls='--', alpha=0.5)
    ax.axvline(result.coarse[1], color='r', ls='--', alpha=0.5)
    ax.set_title(f'Stage 1 — Coarse centroid\n({result.coarse[0]:.1f}, {result.coarse[1]:.1f})')
    ax.set_xlabel('Knife position (px)'); ax.legend()

    # 2. Fine map
    ax = axes[0, 1]
    o = result.fine_offsets
    im = ax.imshow(result.fine_results, extent=[o[0], o[-1], o[-1], o[0]], cmap='viridis')
    fig.colorbar(im, ax=ax, label='+1-order intensity')
    dx_fit = result.fine[0] - result.coarse[0]
    dy_fit = result.fine[1] - result.coarse[1]
    ax.plot(dx_fit, dy_fit, 'rx', markersize=14)
    ax.set_title('Stage 2 — Fine + sub-pixel fit')
    ax.set_xlabel('Δx (px)'); ax.set_ylabel('Δy (px)')

    # 3. Beam size curve + fit
    ax = axes[0, 2]
    R = result.radii
    I = result.radius_intensity
    ax.plot(R, I / I.max(), 'go', label='I(R) data')
    if np.isfinite(result.w0):
        Rfit = np.linspace(R[0], R[-1], 200)
        Ifit = 1.0 - np.exp(-2.0 * Rfit ** 2 / result.w0 ** 2)
        ax.plot(Rfit, Ifit, 'g-', alpha=0.6, label=f'fit w={result.w0:.1f}')
    ax.axvline(result.w0, color='m', ls='--', label='1/e² radius')
    ax.axhline(1 - 1 / np.e ** 2, color='k', ls=':', alpha=0.5)
    ax.set_xlabel('Mask radius R (px)'); ax.set_ylabel('Normalised I(R)')
    ax.set_title('Stage 3 — Beam-size scan')
    ax.legend()

    # 4. Beam plane overlay
    ax = axes[1, 0]
    ax.imshow(np.abs(E_in) ** 2, cmap='gray', origin='upper', aspect='auto')
    ax.plot(cx, cy, 'r+', markersize=18, label='True')
    ax.plot(*result.coarse, 'yx', markersize=10, label='Coarse')
    ax.plot(*result.fine, 'co', mfc='none', markersize=8, label='Final')
    if np.isfinite(result.w0):
        circ = plt.Circle(result.fine, result.w0, color='m', fill=False, lw=1.5,
                          label=f'1/e² circle w={result.w0:.0f}')
        ax.add_patch(circ)
    ax.set_title(f'SLM plane ({M}×{N})'); ax.legend(loc='upper right', fontsize=8)

    # 5. Empty / centre comparison
    ax = axes[1, 1]; ax.axis('off')
    err_c = np.hypot(*(np.subtract(result.coarse, true_center)))
    err_f = np.hypot(*(np.subtract(result.fine, true_center)))
    ax.text(0.0, 0.92, 'Centre report', fontsize=14, weight='bold')
    ax.text(0.0, 0.78, f'true       : ({cx:.2f}, {cy:.2f})', family='monospace')
    ax.text(0.0, 0.68, f'coarse     : ({result.coarse[0]:.2f}, {result.coarse[1]:.2f})', family='monospace')
    ax.text(0.0, 0.58, f'final      : ({result.fine[0]:.2f}, {result.fine[1]:.2f})', family='monospace')
    ax.text(0.0, 0.42, f'err coarse : {err_c:.3f} px', family='monospace', color='orange')
    ax.text(0.0, 0.32, f'err final  : {err_f:.3f} px', family='monospace', color='green')

    # 6. Beam-size comparison
    ax = axes[1, 2]; ax.axis('off')
    ax.text(0.0, 0.92, 'Beam-size report', fontsize=14, weight='bold')
    ax.text(0.0, 0.78, f'true w₀     : {true_w0:.2f} px (1/e² intensity)', family='monospace')
    ax.text(0.0, 0.68, f'measured w₀ : {result.w0:.2f} px', family='monospace')
    err_w = abs(result.w0 - true_w0)
    rel_w = err_w / true_w0 * 100 if true_w0 else float('nan')
    ax.text(0.0, 0.52, f'abs error   : {err_w:.3f} px', family='monospace', color='green')
    ax.text(0.0, 0.42, f'rel error   : {rel_w:.3f} %', family='monospace', color='green')
    ax.text(0.0, 0.22, f'true diam.  : {2*true_w0:.0f} px', family='monospace', color='gray')
    ax.text(0.0, 0.12, f'meas diam.  : {2*result.w0:.0f} px', family='monospace', color='gray')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    E_in = gaussian_beam(TRUE_CENTER, w0=TRUE_W0)
    res = calibrate(E_in)
    err_c = np.hypot(*(np.subtract(res.coarse, TRUE_CENTER)))
    err_f = np.hypot(*(np.subtract(res.fine, TRUE_CENTER)))
    print(f'true centre  = {TRUE_CENTER}')
    print(f'coarse       = ({res.coarse[0]:.3f}, {res.coarse[1]:.3f})  err {err_c:.4f} px')
    print(f'fine         = ({res.fine[0]:.3f}, {res.fine[1]:.3f})  err {err_f:.4f} px')
    print(f'true w0      = {TRUE_W0}')
    print(f'measured w0  = {res.w0:.3f}  err {abs(res.w0 - TRUE_W0):.4f} px')
    _plot(res, TRUE_CENTER, TRUE_W0, E_in)
