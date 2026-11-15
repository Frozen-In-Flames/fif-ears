"""Audio loading.

Decodes any audio file to float samples via ffmpeg (a factual format
dependency, present on the target machine). WAV files could be read
natively, but one decode path for every format keeps behavior uniform
and deterministic.

Returns mono plus the original channel pair so stereo width can be
measured honestly. A mono source decoded to two channels yields two
identical channels, which correctly measures as zero width.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass

import numpy as np


@dataclass
class AudioData:
    mono: np.ndarray          # float32, shape (n,), range about [-1, 1]
    left: np.ndarray          # float32, shape (n,)
    right: np.ndarray         # float32, shape (n,)
    sample_rate: int
    channels: int             # channels in the source file
    duration: float           # seconds

    @property
    def is_stereo(self) -> bool:
        return self.channels >= 2 and not np.array_equal(self.left, self.right)


def _probe(path: str) -> tuple[int, int]:
    """Return (sample_rate, channels) of the first audio stream."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate,channels",
         "-of", "json", path],
        capture_output=True, text=True, check=True,
    )
    streams = json.loads(out.stdout).get("streams", [])
    if not streams:
        raise ValueError(f"no audio stream found in {path!r}")
    return int(streams[0]["sample_rate"]), int(streams[0]["channels"])


def load_audio(path: str) -> AudioData:
    """Decode `path` to float32 samples at its native sample rate."""
    sample_rate, channels = _probe(path)
    # Decode mono sources as mono: an ffmpeg mono-to-stereo upmix applies
    # a pan law that costs 3 dB and would corrupt every level measurement.
    ac = 1 if channels == 1 else 2
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path,
         "-f", "f32le", "-acodec", "pcm_f32le", "-ac", str(ac), "-"],
        capture_output=True, check=True,
    )
    raw = np.frombuffer(proc.stdout, dtype=np.float32)
    if raw.size == 0:
        raise ValueError(f"decoded zero samples from {path!r}")
    if ac == 1:
        left = np.ascontiguousarray(raw)
        right = np.ascontiguousarray(raw)
    else:
        raw = raw[: raw.size - (raw.size % 2)]
        stereo = raw.reshape(-1, 2)
        left = np.ascontiguousarray(stereo[:, 0])
        right = np.ascontiguousarray(stereo[:, 1])
    mono = (left.astype(np.float64) + right.astype(np.float64)) * 0.5
    mono = mono.astype(np.float32)
    return AudioData(
        mono=mono,
        left=left,
        right=right,
        sample_rate=sample_rate,
        channels=channels,
        duration=len(mono) / sample_rate,
    )
