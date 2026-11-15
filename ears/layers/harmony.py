"""Layer 3, pitch and harmony.

Dominant pitch track by YIN (ported approach, see ears/dsp.py), chroma
from the STFT, key by Krumhansl-Schmuckler profile correlation, and a
chord per structure segment by chroma template matching. Chord
confidence is the margin over the runner-up template, so ambiguous
input (noise, silence residue) reports low confidence instead of a
confident wrong answer.
"""

from __future__ import annotations

import numpy as np

from ..audio import AudioData
from ..dsp import (NOTE_NAMES, chroma, estimate_key, freq_to_midi,
                  match_chord, midi_to_name, stft_mag, yin_pitch)

WIN = 2048
HOP = 512
CHORD_MIN_SIM = 0.5
CHORD_MIN_MARGIN = 0.03


def _pitch_track(audio: AudioData):
    x = audio.mono
    sr = audio.sample_rate
    frames = []
    n_frames = 1 + max(0, (len(x) - WIN) // HOP)
    if len(x) < WIN:
        n_frames = 1
    for i in range(n_frames):
        seg = x[i * HOP: i * HOP + WIN]
        if len(seg) < WIN:
            seg = np.pad(seg, (0, WIN - len(seg)))
        freq, conf = yin_pitch(seg.astype(np.float64), sr)
        t = (i * HOP + WIN / 2) / sr
        if freq is not None:
            frames.append({"t": round(float(t), 3), "freq": round(float(freq), 2),
                           "midi": round(float(freq_to_midi(freq)), 2),
                           "note": midi_to_name(freq_to_midi(freq)),
                           "confidence": round(float(conf), 3)})
    return frames


def analyze(audio: AudioData, segments=None) -> dict:
    sr = audio.sample_rate
    track = _pitch_track(audio)
    mags, freqs, times = stft_mag(audio.mono, sr, win=WIN, hop=HOP)
    chr_frames = chroma(mags, freqs)
    frame_energy = mags.astype(np.float64).sum(axis=1)
    active = frame_energy > 0.05 * frame_energy.max() if frame_energy.max() > 0 \
        else np.zeros(len(frame_energy), dtype=bool)
    if np.any(active):
        mean_chroma = chr_frames[active].mean(axis=0)
    else:
        mean_chroma = np.zeros(12)
    root, mode, corr = estimate_key(mean_chroma)
    key_name = f"{NOTE_NAMES[root]} {mode}" if mean_chroma.sum() > 0 else None

    if segments is None:
        segments = [{"start": 0.0, "end": round(audio.duration, 3)}]
    seg_out = []
    for seg in segments:
        mask = (times >= seg["start"]) & (times < seg["end"]) & active
        if np.any(mask):
            seg_chroma = chr_frames[mask].mean(axis=0)
            c_root, c_mode, sim, margin = match_chord(seg_chroma)
        else:
            c_root, c_mode, sim, margin = None, None, 0.0, 0.0
        if c_root is not None and sim >= CHORD_MIN_SIM and margin >= CHORD_MIN_MARGIN:
            chord = f"{NOTE_NAMES[c_root]} {c_mode}"
        else:
            chord = None
        seg_out.append({
            "start": seg["start"], "end": seg["end"],
            "chord": chord,
            "chord_root": NOTE_NAMES[c_root] if c_root is not None else None,
            "chord_mode": c_mode,
            "chord_similarity": round(float(sim), 3),
            "chord_confidence": round(float(margin), 3),
        })

    if track:
        freqs_voiced = np.array([f["freq"] for f in track])
        dominant = float(np.median(freqs_voiced))
        dominant_info = {
            "freq": round(dominant, 2),
            "midi": round(float(freq_to_midi(dominant)), 2),
            "note": midi_to_name(freq_to_midi(dominant)),
        }
    else:
        dominant_info = None
    total_frames = 1 + max(0, (len(audio.mono) - WIN) // HOP)
    return {
        "status": "ok",
        "pitch_track": track,
        "voiced_fraction": round(len(track) / max(1, total_frames), 3),
        "dominant_pitch": dominant_info,
        "key": key_name,
        "key_root": NOTE_NAMES[root] if key_name else None,
        "key_mode": mode if key_name else None,
        "key_confidence": round(float(corr), 3) if key_name else 0.0,
        "mean_chroma": [round(float(v), 4) for v in mean_chroma],
        "segments": seg_out,
    }
