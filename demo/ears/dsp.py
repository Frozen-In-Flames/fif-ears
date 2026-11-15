"""Shared deterministic DSP primitives: STFT features, YIN pitch, chroma.

Everything here is numpy/scipy only, no randomness, no downloads.
"""

from __future__ import annotations

import numpy as np

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Frequency band edges (Hz) shared by several layers.
LOW_BAND = (20.0, 250.0)
MID_BAND = (250.0, 4000.0)
HIGH_BAND = (4000.0, 20000.0)
SPEECH_BAND = (300.0, 3400.0)


def stft_mag(x: np.ndarray, sr: int, win: int = 2048, hop: int = 512):
    """Magnitude STFT with a Hann window.

    Returns (mag, freqs, times): mag shape (frames, bins), float32.
    """
    window = np.hanning(win).astype(np.float64)
    n_frames = 1 + max(0, (len(x) - win) // hop)
    if len(x) < win:
        n_frames = 1
    mags = np.empty((n_frames, win // 2 + 1), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop
        frame = np.zeros(win, dtype=np.float64)
        seg = x[start:start + win]
        frame[:len(seg)] = seg
        spec = np.fft.rfft(frame * window)
        mags[i] = np.abs(spec).astype(np.float32)
    freqs = np.fft.rfftfreq(win, 1.0 / sr)
    times = (np.arange(n_frames) * hop + win / 2) / sr
    return mags, freqs, times


def frame_rms_db(x: np.ndarray, sr: int, win_s: float = 0.1, hop_s: float = 0.05):
    """Short-window RMS level in dBFS. Returns (times, db). Silence floors at -80."""
    win = max(1, int(win_s * sr))
    hop = max(1, int(hop_s * sr))
    n = 1 + max(0, (len(x) - win) // hop)
    if len(x) < win:
        n = 1
    db = np.empty(n, dtype=np.float64)
    times = np.empty(n, dtype=np.float64)
    for i in range(n):
        seg = x[i * hop: i * hop + win].astype(np.float64)
        rms = float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0
        db[i] = 20.0 * np.log10(rms) if rms > 0 else -80.0
        times[i] = (i * hop + win / 2) / sr
    return times, np.maximum(db, -80.0)


def spectral_features(mags: np.ndarray, freqs: np.ndarray):
    """Per-frame spectral descriptors from a magnitude STFT.

    Returns dict of float64 arrays: centroid (Hz), bandwidth (Hz),
    rolloff (Hz, 85 percent), flatness (0..1), zcr-free band shares:
    low_share, mid_share, high_share, speech_share (energy fractions).
    """
    mags64 = mags.astype(np.float64)
    total = mags64.sum(axis=1)
    safe_total = np.where(total > 0, total, 1.0)
    centroid = (mags64 * freqs).sum(axis=1) / safe_total
    diff = freqs - centroid[:, None]
    bandwidth = np.sqrt((mags64 * diff ** 2).sum(axis=1) / safe_total)
    # Rolloff: frequency below which 85 percent of magnitude sits.
    csum = np.cumsum(mags64, axis=1)
    thresh = 0.85 * safe_total
    roll_idx = (csum >= thresh[:, None]).argmax(axis=1)
    rolloff = freqs[roll_idx]
    # Flatness: geometric mean / arithmetic mean of the power spectrum.
    power = mags64 ** 2 + 1e-20
    geo = np.exp(np.mean(np.log(power), axis=1))
    arith = np.mean(power, axis=1)
    flatness = np.where(arith > 0, geo / arith, 0.0)

    def band_share(lo: float, hi: float) -> np.ndarray:
        mask = (freqs >= lo) & (freqs < hi)
        energy = (mags64[:, mask] ** 2).sum(axis=1)
        total_e = (mags64 ** 2).sum(axis=1)
        return np.where(total_e > 0, energy / np.where(total_e > 0, total_e, 1.0), 0.0)

    return {
        "centroid": centroid,
        "bandwidth": bandwidth,
        "rolloff": rolloff,
        "flatness": np.clip(flatness, 0.0, 1.0),
        "low_share": band_share(*LOW_BAND),
        "mid_share": band_share(*MID_BAND),
        "high_share": band_share(*HIGH_BAND),
        "speech_share": band_share(*SPEECH_BAND),
    }


def spectral_flux(mags: np.ndarray) -> np.ndarray:
    """Positive spectral change per frame (onset strength before peak picking)."""
    diffs = np.diff(mags.astype(np.float64), axis=0)
    flux = np.maximum(diffs, 0.0).sum(axis=1)
    return np.concatenate([[0.0], flux])


def zero_crossing_rate(x: np.ndarray, win: int = 2048, hop: int = 512) -> np.ndarray:
    n_frames = 1 + max(0, (len(x) - win) // hop)
    if len(x) < win:
        n_frames = 1
    out = np.zeros(n_frames, dtype=np.float64)
    for i in range(n_frames):
        seg = x[i * hop: i * hop + win]
        if len(seg) > 1:
            out[i] = float(np.mean(np.abs(np.diff(np.signbit(seg)))))
    return out


# ---------------------------------------------------------------------------
# YIN pitch estimation.
#
# Python port of the YIN approach used in the Frozen In Flames
# midi-vocal-trainer (src/audioToMidi.ts, yinPitch): difference function
# from the autocorrelation, cumulative mean normalized difference,
# absolute threshold, parabolic interpolation.
# ---------------------------------------------------------------------------

def yin_pitch(frame: np.ndarray, sr: int, fmin: float = 50.0,
              fmax: float = 1000.0, threshold: float = 0.1):
    """Estimate one frame's pitch. Returns (freq_hz, confidence) or (None, 0.0)."""
    n = len(frame)
    if n < 4 or not np.any(frame):
        return None, 0.0
    size = 1 << (2 * n - 1).bit_length()
    spec = np.fft.rfft(frame.astype(np.float64), size)
    acf = np.fft.irfft(spec * np.conj(spec))[:n]
    if acf[0] <= 0:
        return None, 0.0
    # Difference function (stationary approximation): d(tau) = 2 * (r(0) - r(tau)).
    d = 2.0 * (acf[0] - acf)
    d[0] = 0.0
    # Cumulative mean normalized difference.
    cmnd = np.ones(n, dtype=np.float64)
    running = np.cumsum(d[1:])
    idx = np.arange(1, n)
    cmnd[1:] = d[1:] * idx / np.where(running > 0, running, 1.0)
    tau_min = max(1, int(sr / fmax))
    tau_max = min(n - 2, int(sr / fmin))
    if tau_max <= tau_min:
        return None, 0.0
    tau = None
    for t in range(tau_min, tau_max):
        if cmnd[t] < threshold:
            while t + 1 < tau_max and cmnd[t + 1] < cmnd[t]:
                t += 1
            tau = t
            break
    if tau is None:
        t = tau_min + int(np.argmin(cmnd[tau_min:tau_max]))
        if cmnd[t] > 0.35:  # no convincing periodicity in this frame
            return None, 0.0
        tau = t
    # Parabolic interpolation around the minimum.
    if 0 < tau < n - 1:
        a, b, c = cmnd[tau - 1], cmnd[tau], cmnd[tau + 1]
        denom = a - 2.0 * b + c
        shift = 0.5 * (a - c) / denom if denom != 0 else 0.0
    else:
        shift = 0.0
    freq = sr / (tau + shift)
    confidence = float(max(0.0, 1.0 - cmnd[tau]))
    return float(freq), confidence


def freq_to_midi(freq: float) -> float:
    return 69.0 + 12.0 * np.log2(freq / 440.0)


def midi_to_name(midi: float) -> str:
    m = int(round(midi))
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"


# ---------------------------------------------------------------------------
# Chroma
# ---------------------------------------------------------------------------

def chroma(mags: np.ndarray, freqs: np.ndarray,
           fmin: float = 50.0, fmax: float = 5000.0) -> np.ndarray:
    """12-bin chroma per frame from a magnitude STFT. Rows sum to 1 where voiced."""
    mask = (freqs >= fmin) & (freqs <= fmax)
    f = freqs[mask]
    midi = np.round(freq_to_midi(f)).astype(int) % 12
    sub = mags[:, mask].astype(np.float64)
    out = np.zeros((mags.shape[0], 12), dtype=np.float64)
    for pc in range(12):
        cols = midi == pc
        if np.any(cols):
            out[:, pc] = sub[:, cols].sum(axis=1)
    totals = out.sum(axis=1, keepdims=True)
    return np.where(totals > 0, out / np.where(totals > 0, totals, 1.0), 0.0)


# Krumhansl-Schmuckler key profiles.
KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                     2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                     2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def estimate_key(mean_chroma: np.ndarray):
    """Return (root_pc, mode, correlation) by Krumhansl-Schmuckler matching."""
    best = (0, "major", -2.0)
    if mean_chroma.sum() <= 0:
        return 0, "major", 0.0
    for mode, profile in (("major", KS_MAJOR), ("minor", KS_MINOR)):
        for root in range(12):
            rotated = np.roll(profile, root)
            corr = float(np.corrcoef(mean_chroma, rotated)[0, 1])
            if corr > best[2]:
                best = (root, mode, corr)
    return best


def chord_templates():
    """24 triad templates (12 major, 12 minor) as unit-sum chroma vectors."""
    templates = {}
    for root in range(12):
        for mode, third in (("major", 4), ("minor", 3)):
            vec = np.full(12, 0.02)
            vec[root] = 1.0
            vec[(root + third) % 12] = 0.8
            vec[(root + 7) % 12] = 0.9
            templates[(root, mode)] = vec / vec.sum()
    return templates


def match_chord(segment_chroma: np.ndarray, templates=None):
    """Best triad for a chroma vector.

    Returns (root_pc, mode, similarity, confidence) where confidence is
    the margin between the best and second-best template similarity.
    A flat chroma (noise, silence residue) matches everything about
    equally, so its margin, and therefore its confidence, is near zero.
    Returns (None, None, 0.0, 0.0) for an all-zero chroma.
    """
    if segment_chroma.sum() <= 0:
        return None, None, 0.0, 0.0
    if templates is None:
        templates = chord_templates()
    vec = segment_chroma / segment_chroma.sum()
    sims = []
    for (root, mode), tmpl in templates.items():
        sim = float(np.dot(vec, tmpl) /
                    (np.linalg.norm(vec) * np.linalg.norm(tmpl)))
        sims.append((sim, root, mode))
    sims.sort(reverse=True)
    best_sim, root, mode = sims[0]
    margin = best_sim - sims[1][0]
    return root, mode, best_sim, float(margin)
