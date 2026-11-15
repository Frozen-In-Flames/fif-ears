"""Layer 1, signal: what the file physically is, in numbers."""

from __future__ import annotations

import numpy as np

from ..audio import AudioData
from ..dsp import spectral_features, stft_mag


def analyze(audio: AudioData) -> dict:
    x = audio.mono.astype(np.float64)
    sr = audio.sample_rate
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    rms = float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0
    clipping = int(np.sum(np.abs(x) >= 0.999)) if len(x) else 0

    mags, freqs, _ = stft_mag(audio.mono, sr)
    feats = spectral_features(mags, freqs)
    # Spectral summaries weighted toward frames that carry energy, so a
    # long quiet tail does not drag the numbers away from the content.
    frame_energy = mags.astype(np.float64).sum(axis=1)
    w = frame_energy / frame_energy.sum() if frame_energy.sum() > 0 else None

    def wmean(a):
        return float(np.average(a, weights=w)) if w is not None else 0.0

    def wstd(a):
        if w is None:
            return 0.0
        m = np.average(a, weights=w)
        return float(np.sqrt(np.average((a - m) ** 2, weights=w)))

    # Stereo width: RMS of the side signal over RMS of the mid signal.
    if audio.channels >= 2:
        left = audio.left.astype(np.float64)
        right = audio.right.astype(np.float64)
        mid = (left + right) * 0.5
        side = (left - right) * 0.5
        mid_rms = float(np.sqrt(np.mean(mid ** 2)))
        side_rms = float(np.sqrt(np.mean(side ** 2)))
        width = float(side_rms / mid_rms) if mid_rms > 0 else 0.0
    else:
        width = 0.0

    return {
        "status": "ok",
        "duration_s": round(audio.duration, 3),
        "sample_rate": sr,
        "channels": audio.channels,
        "peak_dbfs": round(20 * np.log10(peak), 2) if peak > 0 else -80.0,
        "rms_dbfs": round(20 * np.log10(rms), 2) if rms > 0 else -80.0,
        "clipping_samples": clipping,
        "spectral_centroid_hz": {"mean": round(wmean(feats["centroid"]), 1),
                                 "std": round(wstd(feats["centroid"]), 1)},
        "spectral_bandwidth_hz": {"mean": round(wmean(feats["bandwidth"]), 1),
                                  "std": round(wstd(feats["bandwidth"]), 1)},
        "spectral_rolloff_hz": {"mean": round(wmean(feats["rolloff"]), 1),
                                "std": round(wstd(feats["rolloff"]), 1)},
        "stereo_width": round(width, 4),
    }
