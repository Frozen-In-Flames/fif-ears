"""Layer 6, events: non-music sound segmentation, classical reference.

Frames are labeled silence, speech-like, music-like, or noise-like from
energy, spectral flatness, and band shares, then merged into segments.
This is a deterministic reference implementation: the labels are
feature-rule categories, not trained-model classifications. The
TrainedEventModel class below defines the adapter interface a trained
model (YAMNet or PANNs style) would implement. Interface only: it is
not an implementation and nothing in this layer calls it.

Rules per frame (hop 512, window 2048), in order:
1. silence: RMS below minus 50 dBFS.
2. noise-like: spectral flatness at or above 0.5 (broadband, no tone).
3. music-like: flatness below 0.08 (strongly tonal).
4. speech-like: at least 50 percent of energy in the 300 to 3400 Hz
   speech band and flatness below 0.5.
5. music-like otherwise (tonal residue).
"""

from __future__ import annotations

import numpy as np

from ..audio import AudioData
from ..dsp import spectral_features, stft_mag

WIN = 2048
HOP = 512
MIN_SEGMENT_S = 0.25


class TrainedEventModel:
    """INTERFACE ONLY, not an implementation.

    A trained audio event model adapter (YAMNet or PANNs style) would
    implement classify_file to return segments in the same shape this
    layer returns: a list of {start, end, label, confidence} dicts
    with model class names as labels. Defined so the classical layer
    can later be swapped or fused without changing the report schema.
    """

    def classify_file(self, path: str):
        raise NotImplementedError(
            "interface only: no trained event model is bundled with fif-ears")


def _frame_labels(audio: AudioData):
    sr = audio.sample_rate
    mags, freqs, times = stft_mag(audio.mono, sr, win=WIN, hop=HOP)
    feats = spectral_features(mags, freqs)
    # Frame RMS in dBFS from the time domain, aligned to the STFT frames.
    x = audio.mono.astype(np.float64)
    rms_db = np.full(len(times), -80.0)
    for i in range(len(times)):
        seg = x[i * HOP: i * HOP + WIN]
        if len(seg):
            r = float(np.sqrt(np.mean(seg ** 2)))
            rms_db[i] = 20 * np.log10(r) if r > 0 else -80.0
    labels = []
    for i in range(len(times)):
        if rms_db[i] < -50.0:
            labels.append("silence")
        elif feats["flatness"][i] >= 0.5:
            labels.append("noise-like")
        elif feats["flatness"][i] < 0.08:
            labels.append("music-like")
        elif feats["speech_share"][i] >= 0.5:
            labels.append("speech-like")
        else:
            labels.append("music-like")
    return times, labels


def _merge(times, labels, duration):
    """Merge consecutive equal labels; absorb runs shorter than MIN_SEGMENT_S."""
    if len(labels) == 0:
        return []
    frame_dur = float(times[1] - times[0]) if len(times) > 1 else duration
    runs = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            runs.append([labels[start], start, i])
            start = i
    # Absorb short runs into the previous run, repeatedly.
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for i, (_, a, b) in enumerate(runs):
            if (b - a) * frame_dur < MIN_SEGMENT_S:
                if i > 0:
                    runs[i - 1][2] = b
                    del runs[i]
                else:
                    runs[1][1] = a
                    del runs[0]
                changed = True
                break
    segments = []
    for label, a, b in runs:
        seg_labels = labels[a:b]
        agree = float(np.mean([1.0 if l == label else 0.0 for l in seg_labels]))
        start_t = float(times[a])
        end_t = float(times[b - 1] + frame_dur) if b - 1 < len(times) else duration
        segments.append({
            "start": round(start_t, 3),
            "end": round(min(end_t, duration), 3),
            "label": label,
            "confidence": round(agree, 3),
        })
    return segments


def analyze(audio: AudioData) -> dict:
    times, labels = _frame_labels(audio)
    segments = _merge(times, labels, audio.duration)
    return {
        "status": "ok",
        "method": "classical feature rules (deterministic reference implementation)",
        "segments": segments,
    }
