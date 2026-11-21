"""Generate the bundled contract fixture, tests/fixtures/contract_mix.wav.

The fixture is synthesized with known ground truth so the contract test
in tests/test_contract_report.py can assert measured numbers, not just
shapes: a C major triad pad (261.63, 329.63, 392.00 Hz) under a click
train at exactly 120 BPM, 8 seconds, mono, 22050 Hz, 16-bit PCM.

This repo bundles no third-party audio (see eval/README.md), so the
fixture used by the fif-smf-1 sidecar contract test is generated here
deterministically and committed as a WAV. Regenerate with:

    python3 tests/fixtures/generate_fixture.py
"""

import os

import numpy as np
from scipy.io import wavfile

SR = 22050
DUR = 8.0


def build():
    t = np.arange(int(DUR * SR)) / SR
    pad = (0.22 * np.sin(2 * np.pi * 261.63 * t)
           + 0.22 * np.sin(2 * np.pi * 329.63 * t)
           + 0.22 * np.sin(2 * np.pi * 392.00 * t))
    out = pad.copy()
    click_s = 0.03
    tt = np.arange(int(click_s * SR)) / SR
    burst = 0.5 * np.sin(2 * np.pi * 2000.0 * tt) * np.exp(-tt / 0.006)
    for k in range(15):  # beats at 0.25 + 0.5k seconds: 120 BPM
        i = int((0.25 + 0.5 * k) * SR)
        out[i:i + len(burst)] += burst[:max(0, len(out) - i)]
    return out


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    pcm = (np.clip(build(), -1, 1) * 32767).astype(np.int16)
    wavfile.write(os.path.join(here, "contract_mix.wav"), SR, pcm)
    print("wrote contract_mix.wav")
