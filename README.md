# fif-ears

Deterministic audio analysis for language models and agents.

A language model has no auditory input channel: presented with an
audio file, it receives a filename, not a sound. fif-ears closes that
gap by measurement. It analyses a file with stated algorithms and
translates the result into structured data that any model can reason
over, with every value reported alongside its confidence, including
honest nulls where the evidence does not support a determination.

## Live demonstration

- [fif-ears in the browser](https://frozen-in-flames.github.io/fif-ears/): the identical analysis package running under Pyodide; an uploaded file is analysed locally and the full report is produced in the page.

Give it an audio file. It returns two files:

- `report.json`: a structured, layered report (schema in SCHEMA.md).
- `report.md`: the same content as a timed narrative written for an AI
  reader: what sounds, where the beats fall, the estimated key and
  chords, where sections change, what the dynamics and groove are
  doing, and the words, if a transcription engine is installed.

The contract is universal and model-agnostic. No dependency on any
agent framework, vendor, or model family. Any agent that can run a
command or read a file can use it.

## The standard

The standard for this tool is to hear enough to use the data. Any
quality of sound that is not yet translated into a measurement is an
open method: researched next, listed openly at the end of every report
and in this README. Nothing here is declared impossible.

One accuracy rule sits underneath that standard and never moves: a
layer reports its real status and never fabricates output. If a layer
cannot run, its status says so and its fields are null. That is an
accuracy rule, not a hearing limit: the translation is missing, the
hearing is the roadmap.

## Install and run

Requirements: Python 3.10 or later, numpy, scipy, and ffmpeg on the
PATH (used to decode audio formats).

```sh
pip install -e .
ears analyze path/to/audio.m4a --out path/to/report-dir
```

Without installing:

```sh
python -m ears analyze path/to/audio.wav --out path/to/report-dir
```

## Role in a fif-smf-1 bundle

Frozen In Flames tools exchange music as fif-smf-1 bundles: a
Standard MIDI File (format 1, PPQ 480, conductor track named "Tempo")
plus a `sidecar.json` manifest. The manifest has an `ears_report`
member, and that member is a fif-ears `report.json` run on the
bundle audio. A consumer reads `summary.tempo_bpm` and `summary.key`
for the headline, then the per-layer results under `layers`, checking
each layer's `status` first: a layer that cannot run reports its real
status and nulls its fields, and the sidecar records `ears_report` as
null when no report was run. The field contract is SCHEMA.md, and
`tests/test_contract_report.py` validates a report produced from the
bundled fixture in `tests/fixtures/` against exactly the fields the
sidecar relies on, including the null shape.

## Layers

| Layer | What it measures | Status |
|---|---|---|
| signal | duration, sample rate, channels, peak and RMS level, clipping, spectral centroid, bandwidth, rolloff, stereo width | IMPLEMENTED (classical) |
| rhythm | onset times, tempo with confidence and alternative metrical levels, beat grid refined against onsets | IMPLEMENTED (classical) |
| structure | novelty curve and section boundaries, ported from the Frozen In Flames Reactor clip-start analysis | IMPLEMENTED (classical, ported) |
| harmony | YIN pitch track, chroma, key estimate, per-section chord with margin-based confidence | IMPLEMENTED (classical) |
| words | speech transcription | IMPLEMENTED when a local Whisper engine is installed (optional dependency: openai-whisper or faster-whisper); otherwise reports status `not_installed` |
| events | sound segments labeled speech-like, music-like, noise-like, silence, from classical feature rules | IMPLEMENTED (classical reference); trained-model adapter is INTERFACE ONLY |
| expression | dynamics contour and slopes, groove and microtiming against the grid, swing, timbre proxies (brightness, warmth, roughness), harmonic tension curve | IMPLEMENTED (classical) |

## What the numbers mean, briefly

- Chord confidence is a margin: how far the best chord template beats
  the runner-up. Flat, ambiguous input (noise) scores near zero and
  reports no chord, instead of a confident wrong answer.
- Tempo is chosen from autocorrelation peaks scored with a standard
  log-Gaussian prior centered at 120 BPM. The report lists the other
  metrical levels measured, so a reader can see the choice that was
  made. The beat grid is then refit against detected onsets.
- Groove is measured in milliseconds against the fitted grid: mean and
  spread of attack deviations, and the mean delay of offbeat onsets
  (the swing number). Positive means behind the grid.
- Harmonic tension runs 0 to 1 per section: circle-of-fifths distance
  of the section chord from the tonic, blended with the chord template
  residual. A clean tonic chord reads near 0.
- Silence is the standing negative control: silence in, nothing out.
  No beats, no notes, no chords, no events beyond silence, flat
  dynamics. The test suite asserts this on every run.

## Tests and verification

Tests are synthesized fixtures with known answers, asserted as
numbers, plus negative and mutation controls (silence, white noise, a
120 BPM fixture that must fail the 130 BPM hypothesis, a reversed
crescendo that must flip its measured slope sign).

```sh
python -m unittest discover -s tests -v
```

Verified at build time (this machine, Python 3.12, numpy 1.26,
scipy 1.11):

- Test suite: 15 of 15 passing in about 4 seconds.
- End-to-end on a real 91.7 second M4A (a performed track whose
  filename states 155 BPM, A Phrygian dominant): fif-ears measured
  151.74 BPM (the 152 BPM metrical level chosen over a stronger slow
  pulse at 44.36 BPM, both listed in the report), dominant pitch D3,
  key estimate D minor at 0.866 correlation, 8 sections with the first
  internal boundary at 2.66 seconds (the same first boundary the
  Reactor analysis finds on this file), 88.4 of 91.7 seconds labeled
  music-like, words layer `not_installed` (no local Whisper on this
  machine).
- Honest gaps in that run: the Krumhansl-Schmuckler key vocabulary
  covers major and minor only. A Phrygian dominant is the fifth mode
  of D harmonic minor, so D minor is the closest answer in that
  vocabulary. Modal key vocabularies are an open method (below).

## Open methods

Translations not yet built, in the order they are being researched:

- Modal and extended key vocabularies (Phrygian dominant, harmonic and
  melodic minor modes, borrowed harmony) beyond major/minor profiles.
- Trained sound-event classification: the events layer defines the
  adapter interface (YAMNet or PANNs style) so a trained model can be
  fused with the classical reference without changing the schema.
- Polyphonic note transcription: an adapter point for models in the
  Basic Pitch family (an ONNX model is already vendored in the Frozen
  In Flames beatbox-to-midi repo) behind the same honesty rules.
- Emotion and felt-quality naming: the expression layer measures the
  correlates (dynamics, groove, timbre, tension). Naming feelings from
  correlates is a translation still being researched.
- Speaker separation and identification for multi-voice recordings.

## Evaluation datasets

Free academic datasets suitable for calibrating each layer are
documented in [eval/README.md](eval/README.md) (LibriSpeech for words,
FSD50K for events, MUSDB18 for music separation and structure, MAESTRO
for piano pitch). They are documented as pointers only: fif-ears
downloads nothing and bundles no third-party data.

## Reuse notes

Built by porting proven Frozen In Flames code rather than reinventing
it: the structure layer is a Python port of Reactor's section-start
analysis, and the pitch layer ports the YIN approach from
midi-vocal-trainer. The rest is new, small, and deterministic:
numpy and scipy only, no randomness, no network, no downloads at
runtime.

## License

Apache-2.0. See LICENSE.

fif-ears is a Frozen In Flames project. Copyright 2026 Aries Hilton.
