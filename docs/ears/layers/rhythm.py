"""Layer 2, rhythm: onset envelope, tempo estimate, beat grid.

Method: spectral flux onset envelope; tempo from the envelope's
autocorrelation over the 40 to 240 BPM lag range; beat phase chosen to
maximize envelope energy on the grid. Silence yields no tempo and no
beats, by construction: a zero envelope has no autocorrelation peak.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from ..audio import AudioData
from ..dsp import spectral_flux, stft_mag

WIN = 2048
HOP = 256
BPM_MIN, BPM_MAX = 40.0, 240.0


def onset_envelope(audio: AudioData):
    mags, _, _ = stft_mag(audio.mono, audio.sample_rate, win=WIN, hop=HOP)
    flux = spectral_flux(mags)
    rate = audio.sample_rate / HOP  # envelope frames per second
    return flux, rate


def _tempo_prior(bpm: float) -> float:
    """Log-Gaussian tempo prior centered on 120 BPM, sigma 0.8 octaves.

    Standard practice in tempo estimation: raw autocorrelation on real
    music often peaks at a slow phrase pulse (a bar or section length)
    rather than the beat. The prior breaks level ties toward the
    metrical level listeners tap to, without overriding a strong
    periodicity: a genuine 60 BPM pulse still wins, because its
    autocorrelation peak is far stronger than the weak half-period
    correlation the prior would boost (proven by the 60 BPM fixture in
    the test suite).
    """
    return float(np.exp(-(np.log2(bpm / 120.0) ** 2) / (2 * 0.8 ** 2)))


def _estimate_tempo(flux: np.ndarray, rate: float):
    """Return (bpm, confidence, candidates) or (None, 0.0, []).

    Candidates are the top autocorrelation peaks scored by
    autocorrelation strength times the tempo prior, so a reader can see
    the metrical levels the estimator chose between.
    """
    env = flux - flux.mean()
    if np.all(env == 0) or float(np.max(flux)) <= 0:
        return None, 0.0, []
    n = len(env)
    size = 1 << (2 * n - 1).bit_length()
    spec = np.fft.rfft(env, size)
    acf = np.fft.irfft(spec * np.conj(spec))[:n]
    if acf[0] <= 0:
        return None, 0.0, []
    lag_min = int(rate * 60.0 / BPM_MAX)
    lag_max = min(n - 1, int(rate * 60.0 / BPM_MIN))
    if lag_max <= lag_min:
        return None, 0.0, []
    window = acf[lag_min:lag_max]
    peaks, _ = find_peaks(window, distance=3)
    scored = []
    if len(peaks):
        for p in peaks:
            lag = lag_min + int(p)
            bpm_raw = 60.0 * rate / lag
            strength = float(acf[lag] / acf[0])
            scored.append((strength * _tempo_prior(bpm_raw), lag, bpm_raw, strength))
    if not scored:
        lag = lag_min + int(np.argmax(window))
        bpm_raw = 60.0 * rate / lag
        strength = float(acf[lag] / acf[0])
        scored = [(strength * _tempo_prior(bpm_raw), lag, bpm_raw, strength)]
    scored.sort(reverse=True)
    _, lag, _, confidence = scored[0]
    # Parabolic refinement of the chosen autocorrelation peak.
    if 0 < lag < n - 1:
        a, b, c = acf[lag - 1], acf[lag], acf[lag + 1]
        denom = a - 2 * b + c
        shift = 0.5 * (a - c) / denom if denom != 0 else 0.0
    else:
        shift = 0.0
    bpm = 60.0 * rate / (lag + shift)
    candidates = [{"bpm": round(s[2], 2), "strength": round(s[3], 3)}
                  for s in scored[:3]]
    return float(bpm), float(confidence), candidates


def _beat_grid(flux: np.ndarray, rate: float, bpm: float, duration: float):
    """Choose the grid phase that captures the most onset energy."""
    period = 60.0 / bpm
    period_frames = period * rate
    best_phase, best_score = 0.0, -1.0
    steps = max(8, int(round(period_frames)))
    for s in range(steps):
        phase = s / rate
        idx = np.arange(phase, duration, period)
        frames = np.clip((idx * rate).astype(int), 0, len(flux) - 1)
        score = float(flux[frames].sum())
        if score > best_score:
            best_score, best_phase = score, phase
    beats = []
    t = best_phase
    while t <= duration:
        beats.append(round(float(t), 3))
        t += period
    return beats, float(best_phase), float(period)


def _refine_onsets(x: np.ndarray, sr: int, approx_times, win_s: float):
    """Snap flux-peak approximations to true attack starts in the time domain.

    A flux peak timestamped at its STFT frame start fires early, as soon
    as the attack enters the analysis window (about 65 ms early with a
    2048 window). The attack start is where the smoothed amplitude first
    crosses 15 percent of the local peak, searched over the window span
    following the approximation. Deterministic, and sharper than the
    frame grid for any percussive onset.
    """
    out = []
    k = max(1, int(0.001 * sr))
    kernel = np.ones(k) / k
    for t in approx_times:
        a = max(0, int((t - 0.01) * sr))
        b = min(len(x), int((t + win_s + 0.02) * sr))
        seg = np.abs(x[a:b].astype(np.float64))
        if len(seg) == 0:
            out.append(round(float(t), 3))
            continue
        sm = np.convolve(seg, kernel, mode="same")
        peak = float(sm.max())
        if peak <= 0:
            out.append(round(float(t), 3))
            continue
        idx = int(np.argmax(sm > 0.15 * peak))
        out.append(round((a + idx) / sr, 3))
    return out


def _refine_grid(onsets, phase, period, duration):
    """Refit phase and period by least squares on grid-inlier onsets,
    then snap each beat to its nearest onset.

    The autocorrelation tempo is accurate but the raw grid phase can sit
    anywhere on the flux plateau the STFT window smears around a sharp
    attack. Onsets are sharper than the plateau, so the grid is refit
    against them. Off-grid onsets (swing, subdivisions) fall outside
    the quarter-period inlier window and cannot pull the fit.
    """
    arr = np.array(onsets, dtype=np.float64)
    if len(arr) >= 4:
        ks = np.round((arr - phase) / period)
        resid = arr - (phase + ks * period)
        inl = np.abs(resid) <= period / 4
        if int(inl.sum()) >= 4:
            k_in, t_in = ks[inl], arr[inl]
            design = np.stack([np.ones_like(k_in), k_in], axis=1)
            coef, *_ = np.linalg.lstsq(design, t_in, rcond=None)
            a, p = float(coef[0]), float(coef[1])
            if p > 0:
                phase, period = a % p, p
    beats = []
    t = phase
    while t <= duration + 1e-9:
        bt = t
        if len(arr):
            j = int(np.argmin(np.abs(arr - t)))
            if abs(float(arr[j]) - t) <= period / 4:
                bt = float(arr[j])
        beats.append(round(float(bt), 3))
        t += period
    return beats, float(phase), float(period)


def analyze(audio: AudioData) -> dict:
    flux, rate = onset_envelope(audio)
    duration = audio.duration
    if float(np.max(flux)) <= 0:
        return {
            "status": "ok",
            "onsets": [],
            "tempo_bpm": None,
            "tempo_confidence": 0.0,
            "tempo_candidates": [],
            "beat_times": [],
            "beat_period_s": None,
            "grid_phase_s": None,
        }
    # Onset peaks: at least 15 percent of the strongest flux, 50 ms apart.
    distance = max(1, int(0.05 * rate))
    peaks, _ = find_peaks(flux, height=0.15 * float(np.max(flux)),
                          distance=distance)
    onsets = _refine_onsets(audio.mono, audio.sample_rate,
                            [float(p / rate) for p in peaks], WIN / audio.sample_rate)
    bpm, confidence, candidates = _estimate_tempo(flux, rate)
    if bpm is None:
        return {
            "status": "ok",
            "onsets": onsets,
            "tempo_bpm": None,
            "tempo_confidence": 0.0,
            "tempo_candidates": [],
            "beat_times": [],
            "beat_period_s": None,
            "grid_phase_s": None,
        }
    _, phase, period = _beat_grid(flux, rate, bpm, duration)
    beats, phase, period = _refine_grid(onsets, phase, period, duration)
    bpm = 60.0 / period
    return {
        "status": "ok",
        "onsets": onsets,
        "tempo_bpm": round(bpm, 2),
        "tempo_confidence": round(confidence, 3),
        "tempo_candidates": candidates,
        "beat_times": beats,
        "beat_period_s": round(period, 4),
        "grid_phase_s": round(phase, 4),
    }
