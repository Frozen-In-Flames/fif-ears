"""Layer 4, structure: novelty curve and section boundaries.

Python port of the section-start analysis in Frozen In Flames Reactor
(index.html, songStarts), the same code that picks clip starts there:

- mono signal decimated by 6, hops of 128 decimated samples
- two cascaded one-pole lowpasses (0.06, then 0.012) split the signal
  into low, mid, and high bands
- novelty per hop is the sum of positive band-RMS changes (one-sided,
  as in the original: a section change registers when it adds band
  energy, which is what a new section starting does)
- novelty is smoothed with a 7-tap moving average (plus/minus 3 hops)
- boundaries are smoothed-novelty peaks at or above 30 percent of the
  maximum, at least 2 seconds apart; time zero is always a boundary

Reactor caps the candidate list at 8 starts for clip picking. Here
max_boundaries defaults to 8 for fidelity with the proven code and is
exposed as a parameter. Files under 8 seconds return only the zero
boundary, as in Reactor.
"""

from __future__ import annotations

import numpy as np

from ..audio import AudioData

DEC = 6
HOP = 128


def novelty_curve(audio: AudioData):
    """Return (times, smoothed_novelty) at the Reactor hop rate."""
    x = audio.mono.astype(np.float64)
    sr = audio.sample_rate
    d = x[::DEC]
    nh = len(d) // HOP
    if nh < 8:
        return np.zeros(0), np.zeros(0)
    nov = np.zeros(nh, dtype=np.float64)
    lp1 = 0.0
    lp2 = 0.0
    prev = np.zeros(3)
    sq = np.zeros(3)
    pos = 0
    for h in range(nh):
        sq[:] = 0.0
        for j in range(HOP):
            v = d[pos]
            pos += 1
            lp1 += 0.06 * (v - lp1)
            lp2 += 0.012 * (lp1 - lp2)
            low = lp2
            mid = lp1 - lp2
            high = v - lp1
            sq[0] += low * low
            sq[1] += mid * mid
            sq[2] += high * high
        bands = np.sqrt(sq / HOP)
        nov[h] = float(np.sum(np.maximum(0.0, bands - prev)))
        prev = bands
    # 7-tap moving average, edges averaged over available taps (as in Reactor).
    sm = np.zeros(nh)
    csum = np.concatenate([[0.0], np.cumsum(nov)])
    for h in range(nh):
        a = max(0, h - 3)
        b = min(nh, h + 4)
        sm[h] = (csum[b] - csum[a]) / (b - a)
    tps = DEC * HOP / sr
    times = np.arange(nh) * tps
    return times, sm


def analyze(audio: AudioData, max_boundaries: int = 8) -> dict:
    duration = audio.duration
    if duration < 8:
        return {
            "status": "ok",
            "boundaries": [0.0],
            "segments": [{"start": 0.0, "end": round(duration, 3)}],
            "novelty": [],
            "note": "file under 8 seconds: single section, as in the Reactor original",
        }
    times, sm = novelty_curve(audio)
    boundaries = [0.0]
    if len(sm) and float(sm.max()) > 0:
        thresh = 0.3 * float(sm.max())
        for h in range(2, len(sm) - 2):
            if sm[h] >= sm[h - 1] and sm[h] >= sm[h + 1] and sm[h] >= thresh:
                t = float(times[h])
                if t <= duration - 1 and all(abs(t - b) >= 2.0 for b in boundaries):
                    boundaries.append(round(t, 2))
    boundaries = boundaries[:max_boundaries]
    segments = []
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else duration
        segments.append({"start": round(float(start), 3),
                         "end": round(float(end), 3)})
    # Downsampled novelty curve for the report (at most 200 points).
    if len(sm):
        step = max(1, len(sm) // 200)
        novelty = [{"t": round(float(times[i]), 2),
                    "v": round(float(sm[i]), 5)}
                   for i in range(0, len(sm), step)]
    else:
        novelty = []
    return {
        "status": "ok",
        "boundaries": [round(float(b), 3) for b in boundaries],
        "segments": segments,
        "novelty": novelty,
    }
