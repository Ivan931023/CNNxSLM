"""
Real-CCD crop & rescale + RMS comparison against the theoretical reference.

Pipeline
--------
1.  Load the raw CCD image (any size, any colour mode) and convert it to a
    background-subtracted single-channel intensity map.  Smoothing is used
    only to *find the centre*, never on values fed into the RMS metrics.
2.  Locate the beam centre with a robust two-stage detector:
        coarse  : weighted centroid of pixels above max(Otsu, 50 % I_max,
                  5 % I_max) on a Gaussian-smoothed copy.  Operates on the
                  largest connected component so a hot pixel can't bias it.
        fine    : 2-D separable parabola through the 3x3 neighbourhood of
                  the smoothed-intensity argmax — sub-pixel.
3.  Decide the side length of the square crop.  Three modes:
        'auto'    -> max(W, H) of the 5%-of-max bounding box * pad.  Uses
                     max(W, H) (not the diagonal) so non-circular shapes
                     such as aberrated flattops, square-symmetry top-hats,
                     or asymmetric lobes fit inside without over-cropping.
        'radius'  -> 2 * R_exp * pad, where R_exp comes from Calibration.html.
        'pixels'  -> a fixed side directly.
    The crop is *always* centred on the detected beam centre and may extend
    past the image edge — the missing pixels are zero-padded so the centre
    stays exactly in the middle of the output.
4.  (Optional but default) Refine the centre by registering the source
    against the theoretical reference, resized to side x side source pixels.
    Phase correlation is computed on the *full intensity* (not the binary
    contour) so it locks onto structural features of an aberrated beam —
    asymmetric lobes, ringing, dark-spot offsets — that a contour-based
    aligner would miss.  3x3 parabolic refinement on the cross-correlation
    peak gives sub-pixel accuracy.
5.  Resample the cropped patch to 500x500 with bicubic interpolation, the
    same pixel grid the CNN expects (src/dataset_generator.py crop_range=250
    -> 500 px).  This 500x500 image is the CNN input — and is *not* used
    for any RMS calculation.
6.  Compute two RMS metrics on the **raw, background-subtracted source pixel
    data** — no smoothing, no rescaling of the experimental image.  The
    theoretical reference is sub-pixel-sampled at source pixel locations
    via the inverse crop transform
        sim_x = 250 + (src_x - cx) * (500 / side)
        sim_y = 250 + (src_y - cy) * (500 / side)
    so the experimental data is touched only by `I_src - bg`.

        RMS_2D  : σ of the raw source intensities over the pixels whose
                  value is >= 95 % of I_src_max.  Pure flatness metric —
                  0 for an ideal flattop, larger for a rippled / Gaussian
                  top.  Reported absolute (peak-normalised) and relative
                  (σ/μ).

        RMS_all : RMS of (I_theo - I_src) over the **intersection of the
                  two 5 % support masks** in source coordinates, both
                  peak-normalised.  Theoretical values come from the
                  bicubic sub-pixel sample at each source pixel — the
                  experimental side is the raw pixel value.

CLI:
    python src/crop_and_rescale.py path/to/ccd.png \
        --reference outputs/images/ideal_flattop.png \
        --out-image outputs/cropped.png --out-json outputs/metrics.json
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, label, zoom


TARGET_SIZE = 500


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
_TURBO_LUT_CACHE: Optional[np.ndarray] = None


def _turbo_lut(n: int = 256) -> np.ndarray:
    """Build (and cache) a (N, 3) RGB lookup table of matplotlib's turbo at
    256 evenly-spaced t values — matches an 8-bit turbo PNG's effective
    t resolution.  Used to invert a colored upload back to linear intensity.
    """
    global _TURBO_LUT_CACHE
    if _TURBO_LUT_CACHE is not None and _TURBO_LUT_CACHE.shape[0] == n:
        return _TURBO_LUT_CACHE
    import matplotlib.cm as cm
    ts = np.linspace(0.0, 1.0, n)
    rgba = cm.turbo(ts)
    _TURBO_LUT_CACHE = (rgba[:, :3] * 255).astype(np.float32)
    return _TURBO_LUT_CACHE


def _is_grayscale_rgb(arr: np.ndarray, sample_n: int = 2500,
                      tol: float = 4.0) -> bool:
    """Quick grayscale check on a (H, W, 3) uint8/float array — sample a
    few thousand pixels and test R≈G≈B."""
    h, w = arr.shape[:2]
    n = min(sample_n, h * w)
    flat = arr.reshape(-1, 3)
    idx = np.linspace(0, flat.shape[0] - 1, n, dtype=np.int64)
    p = flat[idx].astype(np.float32)
    return bool(np.max(np.abs(p[:, 0] - p[:, 1])) <= tol and
                np.max(np.abs(p[:, 1] - p[:, 2])) <= tol)


def _invert_turbo(arr: np.ndarray) -> np.ndarray:
    """Map a (H, W, 3) uint8 RGB image back to a (H, W) float32 linear
    intensity in [0, 1] by nearest-neighbour lookup against the turbo LUT.
    """
    lut = _turbo_lut(256)                               # (N, 3)
    flat = arr.reshape(-1, 3).astype(np.float32)        # (P, 3)
    # Squared Euclidean distance from every pixel to every LUT entry.
    # Memory: P × N × 4 bytes — 250 MB for a 4 K image.  For larger inputs
    # we chunk to keep peak memory bounded.
    out = np.empty(flat.shape[0], dtype=np.float32)
    chunk = 65536
    for s in range(0, flat.shape[0], chunk):
        e = min(s + chunk, flat.shape[0])
        d2 = np.sum((flat[s:e, None, :] - lut[None, :, :]) ** 2, axis=-1)
        out[s:e] = np.argmin(d2, axis=-1)
    out = out / (lut.shape[0] - 1)
    return out.reshape(arr.shape[:2]).astype(np.float32)


def load_intensity(path: str) -> np.ndarray:
    """Load an image as a float intensity map in [0, 1].

    For colored images, assume the matplotlib turbo colormap and invert it
    back to linear intensity (so a flattop's plateau correctly maps to ~1
    and the bg to ~0).  For grayscale images, use luma directly.

    Why: dataset_generator.py renders simulation outputs with cmap='turbo',
    in which the *peak* intensity becomes dark red (low luma) and *mid*
    intensities become bright yellow (high luma).  Computing flatness or
    contour metrics on the raw luma of such a PNG measures variation in
    the bar edges, not the plateau.
    """
    img = Image.open(path)
    if img.mode in ('RGB', 'RGBA'):
        arr = np.asarray(img.convert('RGB'), dtype=np.uint8)
        if _is_grayscale_rgb(arr):
            I = arr.astype(np.float32).mean(axis=-1) / 255.0  # close to luma for gray
        else:
            I = _invert_turbo(arr)
    else:
        I = np.asarray(img.convert('L'), dtype=np.float32) / 255.0
    return I


def estimate_background(I: np.ndarray, border_frac: float = 0.04) -> float:
    """Median of a thin border ring — robust against the beam itself."""
    h, w = I.shape
    b = max(1, int(round(min(h, w) * border_frac)))
    border = np.concatenate([
        I[:b, :].ravel(), I[-b:, :].ravel(),
        I[:, :b].ravel(), I[:, -b:].ravel(),
    ])
    return float(np.median(border))


def background_subtracted(I: np.ndarray) -> np.ndarray:
    bg = estimate_background(I)
    out = I - bg
    np.clip(out, 0.0, None, out=out)
    return out


# ---------------------------------------------------------------------------
# Beam-centre detection
# ---------------------------------------------------------------------------
@dataclass
class BeamCentre:
    cx: float
    cy: float
    radius_5pct: float          # half the bounding-box diagonal (legacy)
    radius_5pct_bbox: float     # half of max(W, H) of the 5% bbox — square-fit half-side
    bbox: Tuple[int, int, int, int]  # x0, y0, x1, y1 of the 5% mask
    method: str


def _otsu_threshold(I: np.ndarray) -> float:
    """Otsu on a 256-bin histogram of [0, max]."""
    Imax = float(I.max())
    if Imax <= 0:
        return 0.0
    hist, edges = np.histogram(I.ravel(), bins=256, range=(0.0, Imax))
    p = hist.astype(np.float64) / max(1, hist.sum())
    omega = np.cumsum(p)
    mu = np.cumsum(p * (edges[:-1] + 0.5 * (edges[1] - edges[0])))
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    denom[denom == 0] = 1e-12
    sigma_b2 = (mu_t * omega - mu) ** 2 / denom
    k = int(np.argmax(sigma_b2))
    return float(edges[k])


def _largest_component(mask: np.ndarray) -> np.ndarray:
    lab, n = label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    keep = int(np.argmax(sizes))
    return lab == keep


def _parabolic_subpixel(I: np.ndarray, cx: int, cy: int) -> Tuple[float, float]:
    """Fit a separable parabola to the 3x3 neighbourhood of (cx, cy)."""
    h, w = I.shape
    if cx <= 0 or cx >= w - 1 or cy <= 0 or cy >= h - 1:
        return float(cx), float(cy)
    ax = I[cy, cx - 1]; bx = I[cy, cx]; cxv = I[cy, cx + 1]
    ay = I[cy - 1, cx]; by = I[cy, cx]; cyv = I[cy + 1, cx]
    dx_den = ax - 2.0 * bx + cxv
    dy_den = ay - 2.0 * by + cyv
    dx = 0.5 * (ax - cxv) / dx_den if dx_den != 0 else 0.0
    dy = 0.5 * (ay - cyv) / dy_den if dy_den != 0 else 0.0
    dx = float(np.clip(dx, -1.0, 1.0))
    dy = float(np.clip(dy, -1.0, 1.0))
    return cx + dx, cy + dy


def detect_centre(I: np.ndarray, smooth_sigma: float = 2.0) -> BeamCentre:
    """Robust beam-centre detector. Returns float pixel coordinates."""
    I_bs = background_subtracted(I)
    Is = gaussian_filter(I_bs, sigma=smooth_sigma)

    Imax = float(Is.max())
    if Imax <= 0:
        h, w = Is.shape
        return BeamCentre(w / 2.0, h / 2.0, min(h, w) / 4.0, min(h, w) / 4.0,
                          (0, 0, w, h), 'fallback-empty')

    t_otsu = _otsu_threshold(Is)
    t_pct = 0.5 * Imax
    threshold = max(t_otsu, t_pct, 0.05 * Imax)

    mask = _largest_component(Is >= threshold)
    if not mask.any():
        mask = Is >= 0.5 * Imax

    ys, xs = np.where(mask)
    weights = Is[mask]
    cx0 = float((xs * weights).sum() / weights.sum())
    cy0 = float((ys * weights).sum() / weights.sum())

    cxi = int(round(cx0)); cyi = int(round(cy0))
    cx_fine, cy_fine = _parabolic_subpixel(Is, cxi, cyi)

    mask5 = _largest_component(Is >= 0.05 * Imax)
    if mask5.any():
        ys5, xs5 = np.where(mask5)
        x0, y0, x1, y1 = int(xs5.min()), int(ys5.min()), int(xs5.max()), int(ys5.max())
        r5_diag = 0.5 * float(np.hypot(x1 - x0, y1 - y0))
        r5_bbox = 0.5 * float(max(x1 - x0, y1 - y0))
    else:
        h, w = Is.shape
        x0 = y0 = 0; x1, y1 = w, h
        r5_diag = 0.25 * min(h, w)
        r5_bbox = 0.25 * min(h, w)

    return BeamCentre(cx_fine, cy_fine, r5_diag, r5_bbox,
                      (x0, y0, x1, y1), 'centroid+parabolic')


# ---------------------------------------------------------------------------
# Crop & rescale
# ---------------------------------------------------------------------------
def crop_centered(I: np.ndarray, cx: float, cy: float, side: float,
                  out_size: Optional[int] = None) -> np.ndarray:
    """Sub-pixel crop+resample: produce an `out_size × out_size` square centred
    at the (sub-pixel) source coordinate (cx, cy), covering `side` source
    pixels.  Bicubic-sampled — no integer-floor rounding artefacts that
    would otherwise leave the source beam offset by up to 1 px from the
    output centre and inflate edge-aligned RMS metrics.

    Out-of-bounds samples are returned as 0 (zero-pad).  When `out_size` is
    omitted, defaults to `round(side)` so the patch keeps a 1:1 source-to-
    patch pixel ratio (useful for template registration).
    """
    from scipy.ndimage import map_coordinates
    if out_size is None:
        out_size = int(round(side))
    h, w = I.shape
    # patch pixel index i corresponds to source coord (pixel-centre convention)
    #   src_x = cx + (i - N/2) * (side / N)
    # so for N == side (1:1 registration patch), patch[N/2] is sub-pixel
    # sampled at exactly src_x = cx — matching the convention used by both
    # detect_centre (centroid is intensity-weighted on integer pixel coords)
    # and resampleTo (identity-on-same-size).  An extra +0.5 inflates RMS_all
    # by ~5 % at sharp edges; verified against the source==ref identity test.
    j = np.arange(out_size, dtype=np.float64)
    src_xs = cx + (j - out_size / 2.0) * (side / out_size)
    src_ys = cy + (j - out_size / 2.0) * (side / out_size)
    yy, xx = np.meshgrid(src_ys, src_xs, indexing='ij')
    out = map_coordinates(I.astype(np.float32, copy=False),
                          [yy.ravel(), xx.ravel()],
                          order=3, mode='constant', cval=0.0,
                          prefilter=True).reshape(out_size, out_size)
    np.clip(out, 0.0, None, out=out)
    return out.astype(np.float32, copy=False)


def rescale_to_target(patch: np.ndarray,
                      target: int = TARGET_SIZE) -> np.ndarray:
    """Cubic resample to `target` x `target`."""
    if patch.shape == (target, target):
        return patch.astype(np.float32, copy=False)
    zy = target / patch.shape[0]
    zx = target / patch.shape[1]
    out = zoom(patch, (zy, zx), order=3, mode='constant', cval=0.0,
               prefilter=True)
    np.clip(out, 0.0, None, out=out)
    if out.shape != (target, target):
        # Numerical edge-case: zoom can return ±1 pixel off — pad/crop.
        fixed = np.zeros((target, target), dtype=np.float32)
        h = min(target, out.shape[0]); w = min(target, out.shape[1])
        fixed[:h, :w] = out[:h, :w]
        out = fixed
    return out.astype(np.float32, copy=False)


def _support_area(I: np.ndarray, frac: float = 0.05) -> int:
    """Pixel count of the largest connected component above frac * I.max()."""
    Imax = float(I.max())
    if Imax <= 0:
        return 0
    return int(_largest_component(I >= frac * Imax).sum())


def crop_and_rescale(I: np.ndarray,
                     mode: str = 'auto',
                     auto_pad: float = 1.10,
                     radius_px: Optional[float] = None,
                     pixels: Optional[int] = None,
                     centre: Optional[BeamCentre] = None,
                     reference_500: Optional[np.ndarray] = None
                     ) -> Tuple[np.ndarray, BeamCentre, float]:
    """Detect the centre, choose a crop side, crop, and resample to 500x500.

    `auto` mode preserves the simulation framing when a reference is given:
        side = 500 * sqrt(A_src_5% / A_ref_5%)
    so the beam ends up occupying the same relative area of the 500x500
    output as it does in the reference.  Without a reference, falls back to
    a tight 5%-bbox crop times `auto_pad` (the beam fills the output).

    Returns (rescaled_500x500, centre_info, crop_side_in_source_px).
    """
    I = I.astype(np.float32, copy=False)
    I_bs = background_subtracted(I)
    if centre is None:
        centre = detect_centre(I_bs)

    if mode == 'pixels':
        if pixels is None or pixels <= 0:
            raise ValueError("mode='pixels' requires a positive `pixels` value")
        side = float(pixels)
    elif mode == 'radius':
        if radius_px is None or radius_px <= 0:
            raise ValueError("mode='radius' requires a positive `radius_px`")
        side = 2.0 * float(radius_px) * float(auto_pad)
    elif mode == 'auto':
        if reference_500 is not None:
            # Preserve the simulation framing: scale source crop so that
            # beam-support area matches the reference's beam-support area
            # at the same relative fraction of the 500x500 frame.
            ref_bs = background_subtracted(reference_500.astype(np.float32))
            # Use a *smoothed* copy purely for support detection (does NOT
            # affect the RMS calculation later).
            ref_smoothed = gaussian_filter(ref_bs, sigma=2.0)
            src_smoothed = gaussian_filter(I_bs, sigma=2.0)
            A_src = _support_area(src_smoothed, frac=0.05)
            A_ref = _support_area(ref_smoothed, frac=0.05)
            if A_src > 0 and A_ref > 0:
                scale = float(np.sqrt(A_src / A_ref))   # src px / ref px
                side = TARGET_SIZE * scale
            else:
                side = 2.0 * centre.radius_5pct_bbox * float(auto_pad)
        else:
            side = 2.0 * centre.radius_5pct_bbox * float(auto_pad)
    else:
        raise ValueError(f"unknown mode: {mode!r}")

    side = max(8.0, side)
    patch = crop_centered(I_bs, centre.cx, centre.cy, side)
    out = rescale_to_target(patch, TARGET_SIZE)
    return out, centre, side


# ---------------------------------------------------------------------------
# Alignment: phase correlation with sub-pixel parabolic refinement
# ---------------------------------------------------------------------------
def phase_correlation_shift(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """Estimate the shift (dx, dy) such that b ~ a shifted by (dx, dy).

    Uses standard phase correlation with a 3x3 parabolic refinement on the
    correlation peak for sub-pixel accuracy.
    """
    A = np.fft.fft2(a)
    B = np.fft.fft2(b)
    R = A * np.conj(B)
    R /= np.maximum(np.abs(R), 1e-12)
    r = np.fft.ifft2(R).real
    h, w = r.shape
    pk = int(np.argmax(r))
    py, px = divmod(pk, w)

    def _val(yy, xx):
        return r[yy % h, xx % w]
    ax = _val(py, px - 1); bx = _val(py, px); cxv = _val(py, px + 1)
    ay = _val(py - 1, px); by = _val(py, px); cyv = _val(py + 1, px)
    dx_den = ax - 2 * bx + cxv
    dy_den = ay - 2 * by + cyv
    dx = 0.5 * (ax - cxv) / dx_den if dx_den != 0 else 0.0
    dy = 0.5 * (ay - cyv) / dy_den if dy_den != 0 else 0.0
    sx = px + dx
    sy = py + dy
    if sx > w / 2: sx -= w
    if sy > h / 2: sy -= h
    return float(sx), float(sy)


def fourier_shift(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    h, w = img.shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    F = np.fft.fft2(img)
    F *= np.exp(-2j * np.pi * (fx * dx + fy * dy))
    out = np.fft.ifft2(F).real
    np.clip(out, 0.0, None, out=out)
    return out.astype(np.float32, copy=False)


def register_to_template(I_src: np.ndarray, theo_500: np.ndarray,
                         cx_init: float, cy_init: float, side: float
                         ) -> Tuple[float, float, Tuple[float, float]]:
    """Refine (cx, cy) in source coordinates by phase correlation against the
    theoretical reference resized to `side x side` source pixels.

    Uses the *full intensity* of both images — captures structural features
    of an aberrated flattop (asymmetric lobes, ringing) that a binary
    contour-based alignment would miss.  Sub-pixel accurate via 3x3
    parabolic refinement of the cross-correlation peak.

    Returns (cx_fine, cy_fine, (dx, dy)) where (dx, dy) is the shift applied
    to the initial guess in source-pixel units.
    """
    src_patch = crop_centered(I_src, cx_init, cy_init, side)
    N = src_patch.shape[0]
    if N < 4:
        return cx_init, cy_init, (0.0, 0.0)

    Ts = theo_500.shape[0]
    z = N / Ts
    theo_resized = zoom(theo_500.astype(np.float32), z, order=3,
                        mode='constant', cval=0.0, prefilter=True)
    if theo_resized.shape != (N, N):
        fixed = np.zeros((N, N), dtype=np.float32)
        h_ = min(N, theo_resized.shape[0]); w_ = min(N, theo_resized.shape[1])
        fixed[:h_, :w_] = theo_resized[:h_, :w_]
        theo_resized = fixed
    np.clip(theo_resized, 0.0, None, out=theo_resized)

    sp = src_patch / max(float(src_patch.max()), 1e-12)
    tp = theo_resized / max(float(theo_resized.max()), 1e-12)

    try:
        # phase_correlation_shift(a, b) returns the shift that aligns b to a.
        # If sp's beam is offset by ε from the patch centre and tp's beam is
        # at the centre, then "shift to apply to tp to align with sp" = +ε.
        # That ε is the correction we want to add to (cx_init, cy_init).
        dx, dy = phase_correlation_shift(sp, tp)
    except Exception:
        dx = dy = 0.0

    cx_fine = cx_init + dx
    cy_fine = cy_init + dy
    return float(cx_fine), float(cy_fine), (float(dx), float(dy))


# ---------------------------------------------------------------------------
# RMS metrics — computed on raw source-pixel data
# ---------------------------------------------------------------------------
@dataclass
class RmsResult:
    rms_2d_abs: float           # σ of raw source pixels in the >=95% region
    rms_2d_rel: float           # σ / μ — flatness ratio
    n_pts_2d: int

    rms_all_abs: float          # RMS of (I_theo - I_src) on the intersected 5% support, in source pixel space
    rms_all_rel: float          # rms_all_abs / mean(I_theo on support)
    n_pts_all: int

    centre_used: Tuple[float, float]
    side_used: float


def _bicubic_sample(I: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sub-pixel bicubic sample of a 2D image at sub-pixel coords (x, y).

    Uses scipy.ndimage.map_coordinates with order=3.  Out-of-bounds → 0.
    """
    from scipy.ndimage import map_coordinates
    coords = np.stack([y.ravel(), x.ravel()], axis=0)
    out = map_coordinates(I, coords, order=3, mode='constant',
                          cval=0.0, prefilter=True)
    return out.reshape(x.shape)


def compute_rms_raw(I_src_bs: np.ndarray, theo_500: np.ndarray,
                    cx: float, cy: float, side: float,
                    cx_ref: Optional[float] = None,
                    cy_ref: Optional[float] = None,
                    top_frac: float = 0.95,
                    contour_frac: float = 0.05
                    ) -> Tuple[RmsResult, np.ndarray, np.ndarray, np.ndarray]:
    """Compute RMS_2D and RMS_all on the **raw, background-subtracted source
    pixel data** — no smoothing, no rescaling of the experimental image.

    The theoretical reference is sub-pixel-sampled at the source pixel
    locations via the inverse crop transform
        sim_x = Ts/2 + (src_x - cx) * (Ts / side)
        sim_y = Ts/2 + (src_y - cy) * (Ts / side)
    so that no resampling of the experimental image is needed.

    Returns:
        rms      : the RmsResult struct
        top_mask : boolean mask in source coords (raw I_src ≥ 95 % I_max)
        supp_mask: boolean mask in source coords for the RMS_all support
        diff_map : |I_theo - I_src_n| in source coords on the support, else 0
    """
    H, W = I_src_bs.shape
    Imax = float(I_src_bs.max())
    if Imax <= 0:
        empty = np.zeros((H, W), dtype=bool)
        return (RmsResult(np.nan, np.nan, 0, np.nan, np.nan, 0, (cx, cy), side),
                empty, empty, np.zeros((H, W), dtype=np.float32))

    src_n = I_src_bs / Imax  # peak-normalised, but values themselves unaltered

    # ---- RMS_2D on RAW source pixels ----------------------------------
    top_mask = src_n >= top_frac
    if top_mask.any():
        vals = src_n[top_mask]
        mean_v = float(vals.mean())
        rms_2d_abs = float(np.sqrt(np.mean((vals - mean_v) ** 2)))
        rms_2d_rel = rms_2d_abs / mean_v if mean_v > 0 else float('nan')
        n2d = int(top_mask.sum())
    else:
        rms_2d_abs = rms_2d_rel = float('nan'); n2d = 0

    # ---- RMS_all on RAW source pixels in the source crop region -------
    # Inverse map: source pixel (x, y) → sim coord
    #   sim_x = cx_ref + (x - cx) * (Ts / side)
    # so the source's beam centroid (at x = cx) lands on the reference's
    # beam centroid (at sim_x = cx_ref).  Defaults to (Ts/2, Ts/2) for a
    # reference whose beam is at the geometric centre.
    Ts = theo_500.shape[0]
    if cx_ref is None: cx_ref = Ts / 2.0
    if cy_ref is None: cy_ref = Ts / 2.0
    scale = Ts / float(side)  # sim px per src px
    half = side / 2.0
    x0 = max(0, int(np.floor(cx - half)) - 1)
    x1 = min(W, int(np.ceil(cx + half)) + 1)
    y0 = max(0, int(np.floor(cy - half)) - 1)
    y1 = min(H, int(np.ceil(cy + half)) + 1)
    supp_mask = np.zeros((H, W), dtype=bool)
    diff_map = np.zeros((H, W), dtype=np.float32)

    if x1 > x0 and y1 > y0:
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        sim_xx = cx_ref + (xx - cx) * scale
        sim_yy = cy_ref + (yy - cy) * scale

        # Inside-the-theoretical-canvas mask
        in_canvas = (sim_xx >= 0) & (sim_xx <= Ts - 1) & \
                    (sim_yy >= 0) & (sim_yy <= Ts - 1)

        theo_max = float(theo_500.max())
        theo_n = theo_500 / max(theo_max, 1e-12)
        theo_at_src = np.zeros_like(xx)
        if in_canvas.any():
            theo_at_src_flat = _bicubic_sample(
                theo_n.astype(np.float32),
                sim_xx[in_canvas], sim_yy[in_canvas],
            )
            theo_at_src[in_canvas] = np.clip(theo_at_src_flat, 0.0, None)

        src_patch_n = src_n[y0:y1, x0:x1]
        in_src  = src_patch_n >= contour_frac
        in_theo = theo_at_src >= contour_frac
        inter   = in_src & in_theo & in_canvas

        if inter.any():
            diff = theo_at_src[inter] - src_patch_n[inter]
            rms_all_abs = float(np.sqrt(np.mean(diff ** 2)))
            denom = float(theo_at_src[inter].mean())
            rms_all_rel = rms_all_abs / denom if denom > 0 else float('nan')
            n_all = int(inter.sum())

            # Visualisation helpers in source coordinates
            supp_mask[y0:y1, x0:x1] = inter
            diff_full = np.zeros_like(xx)
            diff_full[inter] = np.abs(theo_at_src[inter] - src_patch_n[inter])
            diff_map[y0:y1, x0:x1] = diff_full
        else:
            rms_all_abs = rms_all_rel = float('nan'); n_all = 0
    else:
        rms_all_abs = rms_all_rel = float('nan'); n_all = 0

    return (RmsResult(rms_2d_abs=rms_2d_abs, rms_2d_rel=rms_2d_rel, n_pts_2d=n2d,
                      rms_all_abs=rms_all_abs, rms_all_rel=rms_all_rel,
                      n_pts_all=n_all,
                      centre_used=(cx, cy), side_used=float(side)),
            top_mask, supp_mask, diff_map)


# ---------------------------------------------------------------------------
# Convenience: full pipeline + saving
# ---------------------------------------------------------------------------
def save_image_uint8(arr: np.ndarray, path: str, cmap: Optional[str] = None) -> None:
    """Save a [0, 1] float array as PNG.  If `cmap` is given, apply matplotlib
    colormap; otherwise save grayscale to match the CNN's expected mode."""
    a = np.clip(arr / max(1e-12, arr.max()), 0.0, 1.0)
    if cmap is None:
        Image.fromarray(np.round(a * 255).astype(np.uint8), mode='L').save(path)
    else:
        import matplotlib.pyplot as plt  # lazy import
        plt.imsave(path, a, cmap=cmap)


def run(input_path: str,
        reference_path: Optional[str] = None,
        mode: str = 'auto',
        radius_px: Optional[float] = None,
        pixels: Optional[int] = None,
        auto_pad: float = 1.10,
        register: bool = True,
        out_image: Optional[str] = None,
        out_json: Optional[str] = None,
        out_aligned: Optional[str] = None) -> dict:
    """Full pipeline. RMS metrics are computed on the raw, background-
    subtracted source pixels — no smoothing, no rescaling — by sub-pixel
    sampling the theoretical reference at source pixel locations.

    The 500x500 CNN-format image is still produced for downstream use, but
    it is *not* used to compute the RMS metrics.
    """
    I = load_intensity(input_path)
    I_bs = background_subtracted(I)
    centre = detect_centre(I_bs)

    # Pre-load the reference once so we can use it both for auto-sizing and
    # for RMS later.
    ref_500_bs = None
    if reference_path:
        ref_raw = load_intensity(reference_path)
        ref_500 = ref_raw if ref_raw.shape == (TARGET_SIZE, TARGET_SIZE) else \
                  rescale_to_target(ref_raw, TARGET_SIZE)
        ref_500_bs = background_subtracted(ref_500)

    # Choose a crop side. For `auto`, match the reference framing if we have
    # one — otherwise fall back to a tight 5%-bbox crop (beam fills output).
    if mode == 'pixels':
        if pixels is None or pixels <= 0:
            raise ValueError("mode='pixels' requires a positive `pixels` value")
        side = float(pixels)
    elif mode == 'radius':
        if radius_px is None or radius_px <= 0:
            raise ValueError("mode='radius' requires a positive `radius_px`")
        side = 2.0 * float(radius_px) * float(auto_pad)
    elif mode == 'auto':
        if ref_500_bs is not None:
            ref_smoothed = gaussian_filter(ref_500_bs, sigma=2.0)
            src_smoothed = gaussian_filter(I_bs, sigma=2.0)
            A_src = _support_area(src_smoothed, frac=0.05)
            A_ref = _support_area(ref_smoothed, frac=0.05)
            if A_src > 0 and A_ref > 0:
                side = TARGET_SIZE * float(np.sqrt(A_src / A_ref))
            else:
                side = 2.0 * centre.radius_5pct_bbox * float(auto_pad)
        else:
            side = 2.0 * centre.radius_5pct_bbox * float(auto_pad)
    else:
        raise ValueError(f"unknown mode: {mode!r}")
    side = max(8.0, side)

    # Crop window is *always* centred on the detected centroid — phase-corr
    # template registration was previously applied but it drifted the centre
    # for shape-mismatched cases (rotation, scaling).  The detector's
    # parabolic sub-pixel refinement is already accurate; reference-centre
    # offset is now handled inside compute_rms_raw via cx_ref-aware inverse
    # mapping.  `register` flag retained for backward-compat but ignored.
    cx_used, cy_used = centre.cx, centre.cy
    register_shift = (0.0, 0.0)
    register_used = False
    _ = register  # silence unused-arg linters

    # Reference beam centroid (for the cx_ref-aware inverse map below).
    cx_ref = cy_ref = TARGET_SIZE / 2.0
    if ref_500_bs is not None:
        ref_centre = detect_centre(ref_500_bs)
        cx_ref, cy_ref = ref_centre.cx, ref_centre.cy

    # Crop + rescale to 500x500 for the CNN (centred on the detected centroid)
    rescaled = crop_centered(I_bs, cx_used, cy_used, side, out_size=TARGET_SIZE)

    report: dict = {
        'input': os.path.abspath(input_path),
        'source_shape': list(I.shape),
        'detected_centre': {'cx': centre.cx, 'cy': centre.cy,
                            'method': centre.method,
                            'radius_5pct': centre.radius_5pct,
                            'radius_5pct_bbox': centre.radius_5pct_bbox,
                            'bbox': list(centre.bbox)},
        'crop_side_src_px': side,
        'centre_used': {'cx': cx_used, 'cy': cy_used,
                        'register_shift_src_px': list(register_shift),
                        'registered_to_template': register_used},
        'output_shape': list(rescaled.shape),
    }

    if ref_500_bs is not None:
        rms, top_mask, supp_mask, diff_map = compute_rms_raw(
            I_bs, ref_500_bs, cx_used, cy_used, side,
            cx_ref=cx_ref, cy_ref=cy_ref)
        report['reference'] = os.path.abspath(reference_path)
        report['rms'] = asdict(rms)
        report['rms']['support_n_src_px'] = int(supp_mask.sum())
        report['rms']['top_n_src_px']     = int(top_mask.sum())
        if out_aligned:
            save_image_uint8(diff_map, out_aligned, cmap='turbo')

    if out_image:
        save_image_uint8(rescaled, out_image, cmap='turbo')
    if out_json:
        with open(out_json, 'w') as f:
            json.dump(report, f, indent=2)

    return report


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('input', help='Path to a real CCD image (PNG/JPG/TIFF).')
    p.add_argument('--reference', help='Theoretical reference image.', default=None)
    p.add_argument('--mode', choices=['auto', 'radius', 'pixels'], default='auto')
    p.add_argument('--radius-px', type=float, default=None,
                   help="Beam radius (px) — used when --mode=radius. "
                        "Take this from Calibration.html (1/e^2 waist).")
    p.add_argument('--pixels', type=int, default=None,
                   help='Crop side in source pixels — used when --mode=pixels.')
    p.add_argument('--auto-pad', type=float, default=1.10,
                   help='Padding factor applied to auto/radius-derived sides.')
    p.add_argument('--out-image', default=None,
                   help='Where to save the 500x500 result (PNG, turbo).')
    p.add_argument('--out-aligned', default=None,
                   help='Where to save the |I_theo - I_src| difference map (PNG, turbo).')
    p.add_argument('--out-json', default=None,
                   help='Where to dump the metric report (JSON).')
    p.add_argument('--no-register', action='store_true',
                   help='Skip template-matching registration and trust the '
                        'centroid/parabolic detector alone.')
    args = p.parse_args()

    report = run(
        input_path=args.input, reference_path=args.reference,
        mode=args.mode, radius_px=args.radius_px, pixels=args.pixels,
        auto_pad=args.auto_pad, register=not args.no_register,
        out_image=args.out_image, out_json=args.out_json,
        out_aligned=args.out_aligned,
    )
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    _cli()
