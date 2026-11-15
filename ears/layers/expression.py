"""Layer 7, expression: measurable correlates of feel.

Computed classically and deterministically from the other layers:

- dynamics contour: short-window RMS in dB over time, per-section and
  global slope in dB per second (positive slope is a crescendo,
  negative a decrescendo), and dynamic range per section.
- groove and microtiming: how far real attacks sit from the perfect
  beat grid, in milliseconds: mean and spread of beat deviations, and
  the mean delay of offbeat (eighth-note) onsets, which is the swing
  measurement. Positive deviation means late, behind the grid.
- timbre proxies over time: brightness (spectral centroid), warmth
  (share of energy below 250 Hz), roughness proxy (normalized spectral
  flux).
- harmonic tension curve: per section, the distance of the section
  chord from the tonic: circle-of-fifths distance of the chord root
  from the key root (0 at the tonic, 1 at the tritone pole), blended
  with the chord template residual (1 minus similarity). Tension is 0
  at a clean tonic chord and rises as harmony moves away and blurs.
"""

from __future__ import annotations

import numpy as np

from ..audio import AudioData
from ..dsp import NOTE_NAMES, frame_rms_db, spectral_features, spectral_flux, stft_mag

WIN = 2048
HOP = 512


def _slope(times: np.ndarray, values: np.ndarray) -> float:
    if len(times) < 2:
        return 0.0
    return float(np.polyfit(times, values, 1)[0])


def _dynamics(audio: AudioData, segments):
    times, db = frame_rms_db(audio.mono, audio.sample_rate,
                             win_s=0.1, hop_s=0.05)
    audible = db > -60.0
    global_slope = _slope(times[audible], db[audible]) if audible.sum() >= 2 else 0.0
    if audible.sum() >= 4:
        dyn_range = float(np.percentile(db[audible], 95) -
                          np.percentile(db[audible], 10))
    else:
        dyn_range = 0.0
    per_section = []
    for seg in segments:
        mask = (times >= seg["start"]) & (times < seg["end"]) & audible
        if mask.sum() >= 2:
            slope = _slope(times[mask], db[mask])
            mean_db = float(np.mean(db[mask]))
            rng = float(np.percentile(db[mask], 95) - np.percentile(db[mask], 10)) \
                if mask.sum() >= 4 else 0.0
        else:
            slope, mean_db, rng = 0.0, -80.0, 0.0
        per_section.append({
            "start": seg["start"], "end": seg["end"],
            "mean_db": round(mean_db, 2),
            "slope_db_per_s": round(slope, 3),
            "dynamic_range_db": round(rng, 2),
        })
    step = max(1, len(db) // 200)
    contour = [{"t": round(float(times[i]), 2), "db": round(float(db[i]), 1)}
               for i in range(0, len(db), step)]
    return {
        "contour": contour,
        "global_slope_db_per_s": round(global_slope, 3),
        "dynamic_range_db": round(dyn_range, 2),
        "per_section": per_section,
    }


def _groove(rhythm: dict):
    """Grid deviation measurements from the rhythm layer's output."""
    empty = {
        "beat_deviation_mean_ms": None,
        "beat_deviation_std_ms": None,
        "offbeat_delay_ms": None,
        "onsets_measured": 0,
    }
    bpm = rhythm.get("tempo_bpm")
    period = rhythm.get("beat_period_s")
    phase = rhythm.get("grid_phase_s")
    onsets = rhythm.get("onsets") or []
    if not bpm or not period or phase is None or len(onsets) < 4:
        return empty
    onsets_arr = np.array(onsets, dtype=np.float64)
    # Beat deviation: for each grid beat, the nearest onset's signed offset.
    duration_end = float(onsets_arr.max()) + period
    grid = np.arange(phase, duration_end, period)
    devs = []
    for g in grid:
        j = int(np.argmin(np.abs(onsets_arr - g)))
        d = float(onsets_arr[j] - g)
        if abs(d) <= period / 4:
            devs.append(d)
    # Offbeat delay: onsets whose nearest subdivision point is the
    # eighth-note offbeat, and how late they sit past it.
    off_delays = []
    for t in onsets_arr:
        r = (t - phase) % period
        dist_beat = min(r, period - r)
        dist_off = abs(r - period / 2)
        if dist_off < dist_beat and dist_off <= period / 4:
            off_delays.append(r - period / 2)
    return {
        "beat_deviation_mean_ms": round(1000 * float(np.mean(devs)), 2) if devs else None,
        "beat_deviation_std_ms": round(1000 * float(np.std(devs)), 2) if devs else None,
        "offbeat_delay_ms": round(1000 * float(np.mean(off_delays)), 2) if off_delays else None,
        "onsets_measured": len(onsets),
    }


def _timbre(audio: AudioData, segments):
    sr = audio.sample_rate
    mags, freqs, times = stft_mag(audio.mono, sr, win=WIN, hop=HOP)
    feats = spectral_features(mags, freqs)
    flux = spectral_flux(mags)
    flux_max = float(flux.max()) if len(flux) else 0.0
    roughness = flux / flux_max if flux_max > 0 else flux
    energy = mags.astype(np.float64).sum(axis=1)
    active = energy > 0.05 * energy.max() if energy.max() > 0 else energy > 0

    def seg_stats(mask):
        if not np.any(mask):
            return {"brightness_hz": 0.0, "warmth": 0.0, "roughness": 0.0}
        return {
            "brightness_hz": round(float(np.mean(feats["centroid"][mask])), 1),
            "warmth": round(float(np.mean(feats["low_share"][mask])), 3),
            "roughness": round(float(np.mean(roughness[mask])), 3),
        }

    per_section = []
    for seg in segments:
        mask = (times >= seg["start"]) & (times < seg["end"]) & active
        per_section.append({"start": seg["start"], "end": seg["end"],
                            **seg_stats(mask)})
    return {"overall": seg_stats(active), "per_section": per_section}


def _tension(harmony: dict):
    key_root = harmony.get("key_root")
    tonic_pc = NOTE_NAMES.index(key_root) if key_root in NOTE_NAMES else None
    curve = []
    for seg in harmony.get("segments", []):
        tension = None
        # Only from a chord the harmony layer actually reported: a
        # sub-threshold best match is not a chord, so it earns no tension.
        if tonic_pc is not None and seg.get("chord") is not None \
                and seg.get("chord_root") in NOTE_NAMES:
            root_pc = NOTE_NAMES.index(seg["chord_root"])
            fifths = (7 * (root_pc - tonic_pc)) % 12
            fifths_dist = min(fifths, 12 - fifths) / 6.0  # 0 tonic .. 1 tritone pole
            residual = 1.0 - float(seg.get("chord_similarity", 0.0))
            tension = round(0.6 * fifths_dist + 0.4 * residual, 3)
        curve.append({"start": seg["start"], "end": seg["end"],
                      "chord": seg.get("chord"), "tension": tension})
    return curve


def analyze(audio: AudioData, rhythm: dict, harmony: dict, segments) -> dict:
    dynamics = _dynamics(audio, segments)
    groove = _groove(rhythm)
    timbre = _timbre(audio, segments)
    tension = _tension(harmony)
    return {
        "status": "ok",
        "dynamics": dynamics,
        "groove": groove,
        "timbre": timbre,
        "tension_curve": tension,
    }
