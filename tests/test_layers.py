"""Ground-truth tests for fif-ears.

Every fixture is synthesized with a known answer, and every assertion
is on numbers: pitch in Hz, tempo in BPM, boundary in seconds, slope
in dB per second, groove in milliseconds. Negative controls: silence
must produce nothing, noise must not earn a confident key or chord.
Mutation controls: a 120 BPM fixture must fail the 130 BPM hypothesis,
and a reversed crescendo must flip the measured slope sign.

Run: python -m unittest discover -s tests -v   (from the repo root)
"""

import os
import sys
import tempfile
import unittest

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ears.report import build_report  # noqa: E402

SR = 22050


def save_wav(path, samples, sr=SR):
    data = np.asarray(samples)
    if data.ndim == 1:
        pcm = np.clip(data, -1, 1)
        pcm = (pcm * 32767).astype(np.int16)
    else:
        pcm = (np.clip(data, -1, 1) * 32767).astype(np.int16)
    wavfile.write(path, sr, pcm)
    return path


def sine(freq, dur, amp=0.5, sr=SR):
    t = np.arange(int(dur * sr)) / sr
    return amp * np.sin(2 * np.pi * freq * t)


def clicks(times, dur, sr=SR, freq=2000.0, click_s=0.03):
    out = np.zeros(int(dur * sr))
    tt = np.arange(int(click_s * sr)) / sr
    burst = np.sin(2 * np.pi * freq * tt) * np.exp(-tt / 0.006)
    for t0 in times:
        i = int(t0 * sr)
        out[i:i + len(burst)] += burst[:max(0, len(out) - i)]
    return out


class ReportCase(unittest.TestCase):
    """Base: build one report per fixture, cached per class."""

    fixture_builders = {}

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="fif-ears-test-")
        cls.reports = {}
        for name, builder in cls.fixture_builders.items():
            path = os.path.join(cls.tmp, name + ".wav")
            save_wav(path, builder())
            cls.reports[name] = build_report(path)

    def layer(self, fixture, layer):
        return self.reports[fixture]["layers"][layer]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def fx_sine():
    return sine(440.0, 2.0)


def fx_clicks120():
    beats = [0.25 + 0.5 * k for k in range(15)]
    return clicks(beats, 8.0)


def fx_clicks60():
    beats = [0.25 + 1.0 * k for k in range(11)]
    return clicks(beats, 12.0)


def fx_triad_c():
    return sine(261.63, 4.0, 0.3) + sine(329.63, 4.0, 0.3) + sine(392.0, 4.0, 0.3)


def fx_triad_am():
    return sine(220.0, 4.0, 0.3) + sine(261.63, 4.0, 0.3) + sine(329.63, 4.0, 0.3)


def fx_two_sections():
    # Section 2 adds band energy (brighter and louder), which is what
    # the one-sided Reactor novelty measures. See structure.py.
    return np.concatenate([sine(220.0, 8.0, 0.2),
                           sine(440.0, 8.0, 0.45) + sine(554.37, 8.0, 0.45)
                           + sine(880.0, 8.0, 0.45)])


def fx_stereo():
    return np.stack([sine(440.0, 2.0), sine(220.0, 2.0)], axis=1)


def fx_mono_control():
    s = sine(440.0, 2.0)
    return np.stack([s, s], axis=1)


def fx_silence():
    return np.zeros(4 * SR)


def fx_noise():
    rng = np.random.default_rng(1234)
    return 0.3 * rng.standard_normal(4 * SR)


def _swing_times(delay_s):
    times = []
    for k in range(15):
        times.append(0.25 + 0.5 * k)
        times.append(0.25 + 0.5 * k + 0.25 + delay_s)
    return sorted(times)


def fx_swing40():
    return clicks(_swing_times(0.040), 8.0)


def fx_straight():
    return clicks(_swing_times(0.0), 8.0)


def _crescendo(reversed_=False):
    dur = 8.0
    t = np.arange(int(dur * SR)) / SR
    db = np.linspace(-24.0, 0.0, len(t))
    if reversed_:
        db = db[::-1]
    amp = 10 ** (db / 20.0)
    return amp * np.sin(2 * np.pi * 220.0 * t)


def fx_crescendo():
    return _crescendo(False)


def fx_decrescendo():
    return _crescendo(True)


ALL_FIXTURES = {
    "sine": fx_sine, "clicks120": fx_clicks120, "clicks60": fx_clicks60,
    "triad_c": fx_triad_c,
    "triad_am": fx_triad_am, "two_sections": fx_two_sections,
    "stereo": fx_stereo, "mono_control": fx_mono_control,
    "silence": fx_silence, "noise": fx_noise, "swing40": fx_swing40,
    "straight": fx_straight, "crescendo": fx_crescendo,
    "decrescendo": fx_decrescendo,
}


class TestAll(ReportCase):
    fixture_builders = ALL_FIXTURES

    # -- signal ------------------------------------------------------
    def test_stereo_width(self):
        w = self.layer("stereo", "signal")["stereo_width"]
        self.assertGreater(w, 0.5)
        w_mono = self.layer("mono_control", "signal")["stereo_width"]
        self.assertLess(w_mono, 0.05)

    def test_signal_numbers(self):
        sig = self.layer("sine", "signal")
        self.assertAlmostEqual(sig["duration_s"], 2.0, delta=0.05)
        self.assertEqual(sig["sample_rate"], SR)
        self.assertEqual(sig["clipping_samples"], 0)
        self.assertGreater(sig["peak_dbfs"], -8.0)
        self.assertLess(sig["peak_dbfs"], -4.0)

    # -- pitch and harmony -------------------------------------------
    def test_sine_pitch(self):
        har = self.layer("sine", "harmony")
        self.assertIsNotNone(har["dominant_pitch"])
        self.assertAlmostEqual(har["dominant_pitch"]["freq"], 440.0, delta=3.0)
        self.assertEqual(har["dominant_pitch"]["note"], "A4")
        self.assertGreater(har["voiced_fraction"], 0.9)

    def test_triad_key_and_chord(self):
        har = self.layer("triad_c", "harmony")
        self.assertEqual(har["key"], "C major")
        self.assertGreater(har["key_confidence"], 0.8)
        seg = har["segments"][0]
        self.assertEqual(seg["chord"], "C major")
        self.assertGreater(seg["chord_confidence"], 0.05)

    def test_minor_triad(self):
        har = self.layer("triad_am", "harmony")
        self.assertEqual(har["key"], "A minor")
        self.assertEqual(har["segments"][0]["chord"], "A minor")

    # -- rhythm --------------------------------------------------------
    def test_click_tempo(self):
        rhy = self.layer("clicks120", "rhythm")
        self.assertIsNotNone(rhy["tempo_bpm"])
        self.assertAlmostEqual(rhy["tempo_bpm"], 120.0, delta=2.5)
        # Mutation control: the same fixture must fail the 130 hypothesis.
        self.assertGreater(abs(rhy["tempo_bpm"] - 130.0), 5.0)
        # Score in the coverage direction: every true beat must have a
        # grid beat on it. (The grid legitimately continues past the
        # last click to the end of the file, so the reverse direction
        # would score the grid's tail against nothing.)
        true_beats = [0.25 + 0.5 * k for k in range(15)]
        errs = []
        for tb in true_beats:
            errs.append(min(abs(b - tb) for b in rhy["beat_times"]))
        self.assertGreaterEqual(len(rhy["beat_times"]), 12)
        self.assertLess(float(np.mean(errs)), 0.030)
        self.assertLess(float(np.max(errs)), 0.060)

    def test_slow_tempo_not_doubled(self):
        # Guards the tempo prior: a genuine 60 BPM pulse must not be
        # reported at its double-time level.
        rhy = self.layer("clicks60", "rhythm")
        self.assertIsNotNone(rhy["tempo_bpm"])
        self.assertAlmostEqual(rhy["tempo_bpm"], 60.0, delta=2.5)

    # -- structure -----------------------------------------------------
    def test_two_section_boundary(self):
        st = self.layer("two_sections", "structure")
        inner = [b for b in st["boundaries"] if b > 0.5]
        self.assertTrue(inner, "expected at least one internal boundary")
        self.assertLess(min(abs(b - 8.0) for b in inner), 1.0)

    # -- negative controls ----------------------------------------------
    def test_silence_produces_nothing(self):
        rhy = self.layer("silence", "rhythm")
        self.assertIsNone(rhy["tempo_bpm"])
        self.assertEqual(rhy["beat_times"], [])
        self.assertEqual(rhy["onsets"], [])
        har = self.layer("silence", "harmony")
        self.assertEqual(har["voiced_fraction"], 0.0)
        self.assertIsNone(har["dominant_pitch"])
        for seg in har["segments"]:
            self.assertIsNone(seg["chord"])
        evt = self.layer("silence", "events")
        self.assertTrue(evt["segments"])
        for seg in evt["segments"]:
            self.assertEqual(seg["label"], "silence")
        exp = self.layer("silence", "expression")
        self.assertLessEqual(exp["dynamics"]["dynamic_range_db"], 1.0)
        self.assertAlmostEqual(exp["dynamics"]["global_slope_db_per_s"],
                               0.0, delta=0.5)
        self.assertIsNone(exp["groove"]["offbeat_delay_ms"])
        self.assertIsNone(exp["groove"]["beat_deviation_mean_ms"])

    def test_noise_earns_no_confident_harmony(self):
        har = self.layer("noise", "harmony")
        # Bar: a real triad scores above 0.8 correlation (asserted in
        # the triad test class); noise must stay under half that.
        self.assertLess(har["key_confidence"], 0.5)
        for seg in har["segments"]:
            self.assertTrue(
                seg["chord"] is None or seg["chord_confidence"] < 0.1,
                f"noise segment got a confident chord: {seg}")

    # -- events ----------------------------------------------------------
    def test_events_music_and_noise(self):
        evt = self.layer("triad_c", "events")
        labels = {s["label"] for s in evt["segments"]}
        self.assertIn("music-like", labels)
        evt_n = self.layer("noise", "events")
        total = sum(s["end"] - s["start"] for s in evt_n["segments"])
        noise_time = sum(s["end"] - s["start"] for s in evt_n["segments"]
                         if s["label"] == "noise-like")
        self.assertGreater(noise_time / total, 0.8)

    # -- words -----------------------------------------------------------
    def test_words_layer_honesty(self):
        wrd = self.layer("sine", "words")
        self.assertIn(wrd["status"], ("ok", "not_installed", "error"))
        if wrd["status"] == "not_installed":
            self.assertIsNone(wrd["transcript"])
            self.assertIn("openai-whisper", wrd["optional_dependencies"])
            self.assertIn("faster-whisper", wrd["optional_dependencies"])

    # -- expression ------------------------------------------------------
    def test_swing_measurement(self):
        exp = self.layer("swing40", "expression")
        self.assertIsNotNone(exp["groove"]["offbeat_delay_ms"])
        self.assertAlmostEqual(exp["groove"]["offbeat_delay_ms"], 40.0,
                               delta=18.0)
        exp0 = self.layer("straight", "expression")
        self.assertIsNotNone(exp0["groove"]["offbeat_delay_ms"])
        self.assertLessEqual(abs(exp0["groove"]["offbeat_delay_ms"]), 18.0)
        self.assertLess(exp0["groove"]["beat_deviation_std_ms"], 20.0)

    def test_crescendo_slope_and_sign_flip(self):
        exp = self.layer("crescendo", "expression")
        slope = exp["dynamics"]["global_slope_db_per_s"]
        self.assertAlmostEqual(slope, 3.0, delta=0.6)
        exp_r = self.layer("decrescendo", "expression")
        slope_r = exp_r["dynamics"]["global_slope_db_per_s"]
        self.assertAlmostEqual(slope_r, -3.0, delta=0.6)
        self.assertLess(slope * slope_r, 0)  # sign flipped

    def test_tension_low_at_tonic(self):
        exp = self.layer("triad_c", "expression")
        tensions = [s["tension"] for s in exp["tension_curve"]
                    if s["tension"] is not None]
        self.assertTrue(tensions)
        self.assertLess(max(tensions), 0.35)


if __name__ == "__main__":
    unittest.main()
