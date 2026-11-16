"""Illumination gating, stream jitter, rolling-shutter, and drift characterization."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .types_and_math import Pose2D, normalize_angle

MIN_USABLE_LUMA = 22.0
MIN_USABLE_CONTRAST = 7.0
MIN_USABLE_SHARPNESS = 8.0
MIN_RS_PAIR_CONF = 0.15


def measure_sharpness_tenengrad(gray: np.ndarray) -> float:
    """Compute mean squared Sobel-like gradient energy on a grayscale image."""
    assert gray.ndim == 2 and gray.shape[0] >= 3 and gray.shape[1] >= 3
    gx = 0.5 * (gray[1:-1, 2:] - gray[1:-1, :-2])
    gy = 0.5 * (gray[2:, 1:-1] - gray[:-2, 1:-1])
    return float(np.mean(gx * gx + gy * gy))


def assess_illumination(
    gray: np.ndarray,
    exposure_ms: Optional[int] = None,
    max_exposure_ms: Optional[int] = None,
    gain: Optional[float] = None,
    max_gain: Optional[float] = None,
) -> Dict[str, Any]:
    """Classify frame illumination/texture and report auto-exposure saturation."""
    assert gray.ndim == 2 and gray.size > 0
    mean_luma = float(np.mean(gray))
    contrast_std = float(np.std(gray))
    p05 = float(np.percentile(gray, 5.0))
    p95 = float(np.percentile(gray, 95.0))
    sharp = measure_sharpness_tenengrad(gray) if min(gray.shape) >= 3 else 0.0
    exp_sat = bool(
        exposure_ms is not None
        and max_exposure_ms
        and exposure_ms >= 0.9 * max_exposure_ms
        and mean_luma < MIN_USABLE_LUMA
    )
    gain_sat = bool(gain is not None and max_gain and gain >= 0.9 * max_gain and mean_luma < MIN_USABLE_LUMA)
    if mean_luma < MIN_USABLE_LUMA or (contrast_std < MIN_USABLE_CONTRAST and mean_luma < 40.0):
        status = "dark_room"
    elif contrast_std < MIN_USABLE_CONTRAST:
        status = "low_texture"
    else:
        status = "usable"
    return {
        "status": status,
        "visual_slam_viable": status == "usable",
        "mean_luma": round(mean_luma, 3),
        "contrast_std": round(contrast_std, 3),
        "dynamic_range": round(p95 - p05, 3),
        "sharpness_tenengrad": round(sharp, 3),
        "saturated_dark_frac": round(float(np.mean(gray < 10.0)), 4),
        "saturated_bright_frac": round(float(np.mean(gray > 245.0)), 4),
        "exposure_saturated": exp_sat or gain_sat,
    }


def measure_stream_timing(
    timestamps_s: Sequence[float], frame_ids: Optional[Sequence[int]] = None
) -> Dict[str, Any]:
    """Compute stream FPS, frame interval jitter (ms), gaps, and duplicate IDs."""
    ts = np.asarray(timestamps_s, dtype=np.float64)
    n = int(ts.size)
    if n < 2:
        return {
            "frame_count": n,
            "duration_s": 0.0,
            "mean_fps": 0.0,
            "mean_dt_ms": 0.0,
            "std_jitter_ms": 0.0,
            "p95_dt_ms": 0.0,
            "max_gap_ms": 0.0,
            "stale_duplicate_frames": 0,
            "dropped_frame_ids_est": 0,
        }
    dt_ms = np.diff(ts) * 1000.0
    pos_dt = dt_ms[dt_ms > 0.0]
    dur = float(ts[-1] - ts[0])
    dups = 0
    dropped = 0
    if frame_ids is not None and len(frame_ids) == n:
        id_diffs = np.diff(np.asarray(frame_ids, dtype=np.int64))
        dups = int(np.sum(id_diffs <= 0))
        dropped = int(np.sum(np.maximum(0, id_diffs - 1)))
    return {
        "frame_count": n,
        "duration_s": round(dur, 4),
        "mean_fps": round((n - 1) / dur, 3) if dur > 0.0 else 0.0,
        "mean_dt_ms": round(float(np.mean(pos_dt)), 3) if pos_dt.size else 0.0,
        "std_jitter_ms": round(float(np.std(pos_dt)), 3) if pos_dt.size else 0.0,
        "p95_dt_ms": round(float(np.percentile(pos_dt, 95.0)), 3) if pos_dt.size else 0.0,
        "max_gap_ms": round(float(np.max(pos_dt)), 3) if pos_dt.size else 0.0,
        "stale_duplicate_frames": dups,
        "dropped_frame_ids_est": dropped,
    }


def _parabolic_subpixel(corr: np.ndarray, idx: int) -> float:
    """Refine discrete correlation peak with 3-point cyclic quadratic fit."""
    n = corr.size
    ym = float(corr[(idx - 1) % n])
    y0 = float(corr[idx])
    yp = float(corr[(idx + 1) % n])
    denom = 2.0 * (ym - 2.0 * y0 + yp)
    if abs(denom) < 1e-9 or denom >= 0.0:
        return 0.0
    return max(-0.5, min(0.5, (ym - yp) / denom))


def _rfft_peak_scale(spec: np.ndarray, n: int) -> float:
    """Compute theoretical constructive-phase peak of irfft(|spec|, n) in [0, inf)."""
    mags = np.abs(spec)
    total = float(mags[0])
    if n % 2 == 0 and mags.size > 2:
        total += float(2.0 * np.sum(mags[1:-1]) + mags[-1])
    elif mags.size > 1:
        total += float(2.0 * np.sum(mags[1:]))
    return max(1e-9, total / float(n))


def phase_correlate_1d_shift(
    sig_a: np.ndarray, sig_b: np.ndarray, max_shift_px: Optional[float] = None
) -> Tuple[float, float]:
    """Estimate subpixel horizontal shift (sig_b shifted right by +delta relative to sig_a)."""
    a = np.asarray(sig_a, dtype=np.float64)
    b = np.asarray(sig_b, dtype=np.float64)
    assert a.ndim == 1 and a.shape == b.shape and a.size >= 8
    n = a.size
    win = np.hanning(n)
    aw, bw = (a - np.mean(a)) * win, (b - np.mean(b)) * win
    cross = np.fft.rfft(bw) * np.conj(np.fft.rfft(aw))
    peak_mag = float(np.max(np.abs(cross))) if cross.size else 0.0
    spec = cross / np.maximum(np.abs(cross), 0.02 * peak_mag + 1e-9)
    corr = np.fft.irfft(spec, n=n) / _rfft_peak_scale(spec, n)
    if max_shift_px is not None:
        m = int(min(n // 2 - 1, max(1, math.ceil(max_shift_px))))
        mask = np.zeros(n, dtype=bool)
        mask[: m + 1], mask[n - m :] = True, True
        corr = np.where(mask, corr, -1.0)
    idx = int(np.argmax(corr))
    sub = _parabolic_subpixel(corr, idx)
    shift = float(idx - n) if idx > n // 2 else float(idx)
    return (shift + sub, float(corr[idx]))


def _pair_readout_estimate_ms(
    prev_f: np.ndarray, cur_f: np.ndarray, d_omega: float, fx: float
) -> Optional[Tuple[float, float]]:
    """Compute top-vs-bottom band differential skew and readout ms for one frame pair."""
    h = cur_f.shape[0]
    top_a = np.mean(prev_f[int(0.15 * h) : int(0.35 * h), :], axis=0)
    top_b = np.mean(cur_f[int(0.15 * h) : int(0.35 * h), :], axis=0)
    bot_a = np.mean(prev_f[int(0.65 * h) : int(0.85 * h), :], axis=0)
    bot_b = np.mean(cur_f[int(0.65 * h) : int(0.85 * h), :], axis=0)
    s_top, c_top = phase_correlate_1d_shift(top_a, top_b)
    s_bot, c_bot = phase_correlate_1d_shift(bot_a, bot_b)
    if c_top < MIN_RS_PAIR_CONF or c_bot < MIN_RS_PAIR_CONF:
        return None
    du_diff = s_bot - s_top
    # Band centers at 0.25H and 0.75H sit 0.50*T_readout apart in row scan time.
    readout_s = du_diff / (fx * d_omega * 0.5)
    return (du_diff, readout_s * 1000.0)


def estimate_rolling_shutter_metrics(
    frames: Sequence[np.ndarray],
    yaw_rates_rad_s: Sequence[float],
    timestamps_s: Sequence[float],
    fx: float,
    min_yaw_delta_rad_s: float = 0.25,
) -> Dict[str, Any]:
    """Estimate vertical readout skew/ms from top/bottom band phase shift under yaw acceleration."""
    n = min(len(frames), len(yaw_rates_rad_s), len(timestamps_s))
    if n < 2 or fx <= 0.0:
        return {"evaluated": False, "reason": "insufficient_frames", "sample_pairs": 0}
    skews_px: List[float] = []
    readouts_ms: List[float] = []
    usable_pairs = 0
    for i in range(1, n):
        dt = float(timestamps_s[i] - timestamps_s[i - 1])
        d_omega = float(yaw_rates_rad_s[i] - yaw_rates_rad_s[i - 1])
        if dt <= 0.004 or dt > 0.35 or abs(d_omega) < min_yaw_delta_rad_s:
            continue
        if not (assess_illumination(frames[i - 1])["visual_slam_viable"] and assess_illumination(frames[i])["visual_slam_viable"]):
            continue
        usable_pairs += 1
        est = _pair_readout_estimate_ms(frames[i - 1], frames[i], d_omega, fx)
        if est is not None:
            skews_px.append(est[0])
            readouts_ms.append(est[1])
    if not readouts_ms:
        reason = "dark_or_static" if usable_pairs == 0 else "low_band_correlation"
        return {"evaluated": False, "reason": reason, "sample_pairs": usable_pairs}
    med_ms = float(np.median(readouts_ms))
    return {
        "evaluated": True,
        "sample_pairs": len(readouts_ms),
        "estimated_readout_ms": round(med_ms, 3),
        "median_abs_skew_px": round(float(np.median(np.abs(skews_px))), 3),
        "readout_fraction_at_15fps": round(med_ms / (1000.0 / 15.0), 4),
    }


def evaluate_visual_vs_odom_drift(
    visual_poses: Sequence[Pose2D],
    wheel_odom_poses: Sequence[Pose2D],
    imu_yaws_rad: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Measure translation RMSE (mm) and heading drift (deg) between trajectories."""
    n = min(len(visual_poses), len(wheel_odom_poses))
    if n == 0:
        return {"sample_count": 0, "trans_rmse_mm": 0.0, "max_trans_err_mm": 0.0, "final_yaw_diff_deg": 0.0}
    sq_errs = [
        (visual_poses[i].x_mm - wheel_odom_poses[i].x_mm) ** 2
        + (visual_poses[i].y_mm - wheel_odom_poses[i].y_mm) ** 2
        for i in range(n)
    ]
    yaw_diffs_deg = [
        math.degrees(abs(normalize_angle(visual_poses[i].theta_rad - wheel_odom_poses[i].theta_rad)))
        for i in range(n)
    ]
    out: Dict[str, Any] = {
        "sample_count": n,
        "trans_rmse_mm": round(math.sqrt(sum(sq_errs) / n), 3),
        "max_trans_err_mm": round(math.sqrt(max(sq_errs)), 3),
        "mean_yaw_diff_deg": round(sum(yaw_diffs_deg) / n, 3),
        "final_yaw_diff_deg": round(yaw_diffs_deg[-1], 3),
    }
    if imu_yaws_rad is not None and len(imu_yaws_rad) >= n:
        imu_errs = [
            math.degrees(abs(normalize_angle(visual_poses[i].theta_rad - float(imu_yaws_rad[i]))))
            for i in range(n)
        ]
        out["final_visual_vs_imu_deg"] = round(imu_errs[-1], 3)
    return out
