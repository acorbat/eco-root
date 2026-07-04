from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk
from scipy import ndimage, signal


@dataclass
class ReverberationFilterParams:
    """Parameters used by the reverberation preprocessing pipeline.

    Mapping to preprocessing stages:
    - Pre-step high-pass normalization: gaussian_sigma
    - Fourier spacing estimation: peak_frequency_cycles_per_pixel, peak_frequency_iqr_cycles_per_pixel, spacing_median_px
    - Autocorrelation period estimation: estimated_period_px
    - Targeted SWT suppression: swt_*
    - Multi-harmonic comb notch: comb_sigma_cycles_per_pixel, comb_harmonics, comb_strength
    - Row-energy sigmoid gate: gate_*
    """

    gaussian_sigma: tuple[float, float] = (8.0, 8.0)
    central_column_fraction: tuple[float, float] = (0.25, 0.75)

    peak_frequency_cycles_per_pixel: float = 0.08
    peak_frequency_iqr_cycles_per_pixel: float = 0.0
    spacing_median_px: float = 13.0

    estimated_period_px: int = 13
    comb_sigma_cycles_per_pixel: float = 0.01
    comb_harmonics: tuple[int, ...] = (1, 2, 3)
    comb_strength: float = 0.97

    gate_z_threshold: float = 1.2
    gate_steepness: float = 4.0
    gate_max_suppression: float = 0.85

    swt_wavelet: str = "sym8"
    swt_levels: int = 5
    swt_min_level: int = 1
    swt_max_level: int = 3
    swt_orders: int = 6
    swt_thresh_sigma: float = 2.5


@dataclass
class ReverberationDiagnostics:
    """Diagnostic arrays produced during parameter characterization."""

    peak_frequency_by_plane: np.ndarray
    spacing_by_plane_px: np.ndarray
    autocorrelation: np.ndarray
    autocorrelation_lags: np.ndarray



def load_dicom_volume(
    dicom_path: str | Path,
    roi_xyxy: tuple[int, int, int, int] | None = None,
    channel_index: int = 0,
) -> tuple[np.ndarray, pydicom.dataset.FileDataset]:
    """Load DICOM cine into a (z, y, x) float32 volume.

    roi_xyxy uses inclusive bounds: (min_x, max_x, min_y, max_y).
    """
    dataset = pydicom.dcmread(str(dicom_path))
    raw = dataset.pixel_array

    if raw.ndim == 4:
        frames = raw[..., channel_index]
    elif raw.ndim == 3:
        frames = raw
    elif raw.ndim == 2:
        frames = raw[np.newaxis, ...]
    else:
        raise ValueError(f"Unsupported DICOM pixel array shape: {raw.shape}")

    frames = frames.astype(np.float32)
    if roi_xyxy is None:
        return frames, dataset

    min_x, max_x, min_y, max_y = roi_xyxy
    y0 = max(0, min_y)
    y1 = min(frames.shape[1] - 1, max_y)
    x0 = max(0, min_x)
    x1 = min(frames.shape[2] - 1, max_x)
    if y0 > y1 or x0 > x1:
        raise ValueError("Invalid ROI after clipping to image bounds.")

    return frames[:, y0 : y1 + 1, x0 : x1 + 1], dataset


def compute_high_pass_normalized_volume(
    volume: np.ndarray,
    gaussian_sigma: tuple[float, float] = (8.0, 8.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pre-step: Gaussian background removal + per-plane z-score normalization.

    Diagram mapping:
    volume (raw cropped ROI)
      -> high_pass_volume (z-score normalized), using sigma=(8, 8).

    Returns:
      high_pass_volume, background, high_pass_mean, high_pass_std
    """
    background = ndimage.gaussian_filter(volume, sigma=(0, gaussian_sigma[0], gaussian_sigma[1]))
    high_pass_raw = volume - background
    hp_mean = high_pass_raw.mean(axis=(1, 2), keepdims=True)
    hp_std = high_pass_raw.std(axis=(1, 2), keepdims=True) + 1e-8
    high_pass = (high_pass_raw - hp_mean) / hp_std
    return high_pass.astype(np.float32), background.astype(np.float32), hp_mean, hp_std


def _central_column_bounds(width: int, central_column_fraction: tuple[float, float]) -> tuple[int, int]:
    """Compute robust [start, end) bounds for the central column window."""
    c0 = int(width * central_column_fraction[0])
    c1 = max(int(width * central_column_fraction[1]), c0 + 1)
    return c0, c1


def estimate_fourier_spacing_parameters(
    high_pass_volume: np.ndarray,
    *,
    central_column_fraction: tuple[float, float] = (0.25, 0.75),
    frequency_band_cycles_per_pixel: tuple[float, float] = (0.005, 0.5),
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Parameter estimation #1: Fourier depth spectrum.

    Estimates peak_frequency_median and spacing_median (= 1/f), plane by plane.

    Returns:
      peak_frequency_median, spacing_median, peak_frequency_by_plane, spacing_by_plane
    """
    if high_pass_volume.ndim != 3:
        raise ValueError(f"Expected (z, y, x), got {high_pass_volume.shape}.")

    _, height, width = high_pass_volume.shape
    c0, c1 = _central_column_bounds(width, central_column_fraction)

    depth_profiles = high_pass_volume[:, :, c0:c1].mean(axis=2)
    depth_profiles = signal.detrend(depth_profiles, axis=1)

    frequency_axis = np.fft.rfftfreq(height, d=1.0)
    power_spectrum_by_plane = np.abs(np.fft.rfft(depth_profiles, axis=1)) ** 2

    f_lo, f_hi = frequency_band_cycles_per_pixel
    valid_mask = (frequency_axis > f_lo) & (frequency_axis < f_hi)
    valid_frequency_axis = frequency_axis[valid_mask]
    valid_power = power_spectrum_by_plane[:, valid_mask]

    peak_idx = np.argmax(valid_power, axis=1)
    peak_frequency_by_plane = valid_frequency_axis[peak_idx]

    spacing_by_plane = np.divide(
        1.0,
        peak_frequency_by_plane,
        out=np.full_like(peak_frequency_by_plane, np.nan, dtype=np.float64),
        where=peak_frequency_by_plane > 0,
    )
    return (
        float(np.median(peak_frequency_by_plane)),
        float(np.nanmedian(spacing_by_plane)),
        peak_frequency_by_plane,
        spacing_by_plane,
    )


def estimate_period_from_y_autocorrelation(
    high_pass_volume: np.ndarray,
    *,
    central_column_fraction: tuple[float, float] = (0.25, 0.75),
    target_period_px: float,
    period_search_half_width: int = 3,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Parameter estimation #2: y-autocorrelation near target period.

    Search window is target_period_px ± period_search_half_width.

    Returns:
      estimated_period_px, lag_axis, autocorrelation
    """
    _, _, width = high_pass_volume.shape
    c0, c1 = _central_column_bounds(width, central_column_fraction)

    y_profile = np.median(high_pass_volume[:, :, c0:c1], axis=(0, 2)).astype(np.float64)
    y_profile = signal.detrend(y_profile)
    y_profile -= y_profile.mean()

    autocorr = signal.correlate(y_profile, y_profile, mode="full")
    autocorr = autocorr[len(y_profile) - 1 :]
    autocorr[0] = 0.0

    lag_axis = np.arange(len(autocorr), dtype=np.int32)
    lo = max(2, int(round(target_period_px)) - period_search_half_width)
    hi = min(len(autocorr) - 1, int(round(target_period_px)) + period_search_half_width)
    if lo <= hi:
        local_lags = lag_axis[lo : hi + 1]
        estimated_period_px = int(local_lags[np.argmax(autocorr[lo : hi + 1])])
    else:
        estimated_period_px = int(round(target_period_px))

    return estimated_period_px, lag_axis, autocorr


def estimate_reverberation_parameters(
    volume: np.ndarray,
    *,
    gaussian_sigma: tuple[float, float] = (8.0, 8.0),
    central_column_fraction: tuple[float, float] = (0.25, 0.75),
    frequency_band_cycles_per_pixel: tuple[float, float] = (0.005, 0.5),
    target_period_px: float | None = None,
    period_search_half_width: int = 3,
) -> tuple[ReverberationFilterParams, ReverberationDiagnostics]:
    """Estimate all parameters needed by the reverberation filtering stack."""
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D (z, y, x) volume, got shape {volume.shape}.")

    high_pass, _, _, _ = compute_high_pass_normalized_volume(volume.astype(np.float32), gaussian_sigma=gaussian_sigma)

    peak_frequency_median, spacing_median, peak_frequency_by_plane, spacing_by_plane = estimate_fourier_spacing_parameters(
        high_pass,
        central_column_fraction=central_column_fraction,
        frequency_band_cycles_per_pixel=frequency_band_cycles_per_pixel,
    )

    if target_period_px is None:
        target_period_px = spacing_median

    estimated_period_px, lag_axis, autocorr = estimate_period_from_y_autocorrelation(
        high_pass,
        central_column_fraction=central_column_fraction,
        target_period_px=target_period_px,
        period_search_half_width=period_search_half_width,
    )

    p25, p75 = np.percentile(peak_frequency_by_plane, [25, 75])
    peak_iqr = float(p75 - p25)

    f0 = 1.0 / max(float(estimated_period_px), 1.0)
    harmonics = np.arange(1, int(np.floor(0.5 / f0)) + 1, dtype=int)
    comb_sigma = max(1e-4, f0 / 5.0)

    params = ReverberationFilterParams(
        gaussian_sigma=gaussian_sigma,
        central_column_fraction=central_column_fraction,
        peak_frequency_cycles_per_pixel=float(peak_frequency_median),
        peak_frequency_iqr_cycles_per_pixel=float(peak_iqr),
        spacing_median_px=float(spacing_median),
        estimated_period_px=int(estimated_period_px),
        comb_sigma_cycles_per_pixel=float(comb_sigma),
        comb_harmonics=tuple(int(h) for h in harmonics.tolist()),
    )

    diagnostics = ReverberationDiagnostics(
        peak_frequency_by_plane=peak_frequency_by_plane,
        spacing_by_plane_px=spacing_by_plane,
        autocorrelation=autocorr,
        autocorrelation_lags=lag_axis,
    )
    return params, diagnostics


def _suppress_reverb_col(
    col: np.ndarray,
    delta_k: int,
    *,
    wavelet: str,
    levels: int,
    min_level: int,
    max_level: int,
    n_orders: int,
    thresh_sigma: float,
    strength: float,
    pulse_half_win: int,
) -> np.ndarray:
    """Apply targeted SWT coefficient suppression to one depth column."""
    try:
        import pywt
    except ImportError as exc:
        raise ImportError("pywt is required for SWT-based reverberation suppression.") from exc

    n_orig = len(col)
    pad_len = int(np.ceil(n_orig / 2**levels)) * 2**levels
    col_f64 = np.pad(col.astype(np.float64), (0, pad_len - n_orig), mode="edge")
    n = len(col_f64)

    swt = pywt.swt(col_f64, wavelet, level=levels, norm=False)
    det = [cD for (_, cD) in swt]
    app = [cA for (cA, _) in swt]

    cleaned_det: list[np.ndarray] = []
    for lvl, cD in enumerate(det):
        if lvl < min_level or lvl > max_level:
            cleaned_det.append(cD)
            continue

        c = cD.copy()
        energy = np.abs(c)
        thr = energy.mean() + thresh_sigma * energy.std()

        above = np.where(energy > thr)[0]
        candidates: list[int] = []
        for pos in above:
            lo = max(0, int(pos) - pulse_half_win)
            hi = min(n, int(pos) + pulse_half_win + 1)
            if energy[pos] >= energy[lo:hi].max():
                candidates.append(int(pos))

        is_reverb = np.zeros(n, dtype=bool)
        for p0 in sorted(candidates):
            if is_reverb[p0]:
                continue
            for m in range(1, n_orders + 1):
                center = p0 + m * delta_k
                if center >= n:
                    break
                win = pulse_half_win + m
                lo = max(0, center - win)
                hi = min(n, center + win + 1)
                gain = 1.0 - strength * np.exp(-0.3 * (m - 1))
                c[lo:hi] *= gain
                is_reverb[lo:hi] = True

        cleaned_det.append(c)

    return pywt.iswt([(app[i], cleaned_det[i]) for i in range(levels)], wavelet)[:n_orig].astype(np.float32)


def apply_targeted_swt_suppression(
    high_pass_volume: np.ndarray,
    params: ReverberationFilterParams,
) -> np.ndarray:
    """Targeted SWT suppression stage.

    Uses:
    - wavelet=sym8, levels=5, active levels=[1..3]
    - n_orders=6, thresh_sigma=2.5
    - period candidates {T-1, T, T+1} with strengths {0.50, 0.90, 0.50}
    - pulse_half_win=clip(round(T/3), 4, 24)
    """
    plane_count, _, image_width = high_pass_volume.shape

    t = max(2, int(params.estimated_period_px))
    period_candidates = sorted({p for p in [t - 1, t, t + 1] if p >= 2})
    period_strength = {p: (0.90 if p == t else 0.50) for p in period_candidates}
    pulse_half_window_px = int(np.clip(round(t / 3.0), 4, 24))

    out = np.empty_like(high_pass_volume, dtype=np.float32)
    for pi in range(plane_count):
        for ci in range(image_width):
            col = high_pass_volume[pi, :, ci]
            for period in period_candidates:
                col = _suppress_reverb_col(
                    col,
                    period,
                    wavelet=params.swt_wavelet,
                    levels=params.swt_levels,
                    min_level=params.swt_min_level,
                    max_level=params.swt_max_level,
                    n_orders=params.swt_orders,
                    thresh_sigma=params.swt_thresh_sigma,
                    strength=period_strength[period],
                    pulse_half_win=pulse_half_window_px,
                )
            out[pi, :, ci] = col
    return out


def apply_multi_harmonic_comb_notch(
    swt_vol_filtered: np.ndarray,
    params: ReverberationFilterParams,
) -> np.ndarray:
    """Multi-harmonic comb notch stage.

    Uses:
    - T0 = estimated_period_px, f0 = 1/T0
    - harmonics k where k*f0 <= 0.5 (Nyquist)
    - comb sigma = f0/5, depth = 0.97

    Returns comb_filtered_vol.
    """
    _, height, _ = swt_vol_filtered.shape
    row_freqs = np.fft.fftshift(np.fft.fftfreq(height, d=1.0))

    f0 = 1.0 / max(float(params.estimated_period_px), 1.0)
    comb_gain = np.ones(height, dtype=np.float64)
    for k in params.comb_harmonics:
        dist = np.abs(np.abs(row_freqs) - k * f0)
        comb_gain *= 1.0 - params.comb_strength * np.exp(-0.5 * (dist / params.comb_sigma_cycles_per_pixel) ** 2)

    fft_volume = np.fft.fftshift(np.fft.fft2(swt_vol_filtered, axes=(1, 2)), axes=(1, 2))
    fft_volume *= comb_gain[:, None][None, :, :]
    return np.fft.ifft2(np.fft.ifftshift(fft_volume, axes=(1, 2)), axes=(1, 2)).real.astype(np.float32)


def apply_row_energy_sigmoid_gate(
    comb_filtered_vol: np.ndarray,
    params: ReverberationFilterParams,
) -> np.ndarray:
    """Row-energy sigmoid gate stage.

    Uses:
    - central columns [25%, 75%]
    - robust row z-score from median/IQR
    - z_threshold=1.2, steepness=4.0, max_suppression=0.85

    Returns final_cleaned_vol.
    """
    _, _, width = comb_filtered_vol.shape
    c0, c1 = _central_column_bounds(width, params.central_column_fraction)

    row_energy = comb_filtered_vol[:, :, c0:c1].mean(axis=2)
    row_med = np.median(row_energy, axis=1, keepdims=True)
    row_iqr = (
        np.percentile(row_energy, 75, axis=1, keepdims=True)
        - np.percentile(row_energy, 25, axis=1, keepdims=True)
        + 1e-8
    )
    row_z = (row_energy - row_med) / (row_iqr * 1.4826)

    gate_x = np.clip(params.gate_steepness * (row_z - params.gate_z_threshold), -500.0, 500.0)
    exp_neg = np.exp(np.minimum(gate_x, 0.0))
    sigmoid = exp_neg / (exp_neg + np.exp(-np.maximum(gate_x, 0.0)))
    suppression_weight = np.clip(
        1.0 - params.gate_max_suppression * sigmoid,
        1.0 - params.gate_max_suppression,
        1.0,
    ).astype(np.float32)

    return (comb_filtered_vol * suppression_weight[:, :, None]).astype(np.float32)


def apply_reverberation_filters(
    volume: np.ndarray,
    params: ReverberationFilterParams,
    *,
    clip_nonnegative: bool = False,
) -> np.ndarray:
    """Apply reverberation filtering with the full preprocessing sequence.

    Sequence:
    high_pass -> targeted SWT -> comb notch -> row-energy gate.

    If clip_nonnegative=True, outputs are clipped to [0, +inf).
    Keep clip_nonnegative=False for closest parity with notebook high-pass-domain results.
    """
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D (z, y, x) volume, got shape {volume.shape}.")

    volume_f32 = volume.astype(np.float32)
    high_pass, background, hp_mean, hp_std = compute_high_pass_normalized_volume(
        volume_f32,
        gaussian_sigma=params.gaussian_sigma,
    )

    swt_vol_filtered = apply_targeted_swt_suppression(high_pass, params)

    comb_filtered_vol = apply_multi_harmonic_comb_notch(swt_vol_filtered, params)

    final_cleaned_vol = apply_row_energy_sigmoid_gate(comb_filtered_vol, params)
    if clip_nonnegative:
        final_cleaned_vol = np.clip(final_cleaned_vol, 0.0, None)

    return final_cleaned_vol.astype(np.float32)


def preprocess_dicom_reverberation(
    dicom_path: str | Path,
    *,
    roi_xyxy: tuple[int, int, int, int] | None = None,
    channel_index: int = 0,
    estimate_kwargs: dict | None = None,
    output_domain: str = "image",
    clip_nonnegative: bool = False,
) -> tuple[np.ndarray, ReverberationFilterParams, ReverberationDiagnostics]:
    """Convenience wrapper: load DICOM, estimate parameters, apply filtering."""
    volume, _ = load_dicom_volume(dicom_path, roi_xyxy=roi_xyxy, channel_index=channel_index)
    params, diagnostics = estimate_reverberation_parameters(volume, **(estimate_kwargs or {}))
    filtered_volume = apply_reverberation_filters(
        volume,
        params,
        output_domain=output_domain,
        clip_nonnegative=clip_nonnegative,
    )
    return filtered_volume, params, diagnostics
