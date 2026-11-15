---
title: fif-ears
emoji: 👂
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
---

# fif-ears: audio perception for agents

Most AI models cannot hear. Hand a model an audio file and it gets a
filename, not a sound. fif-ears closes that gap: it listens to a file
as measurement and translates what is in the audio into structured
data any human or model can reason over.

Upload an audio file (wav, mp3, m4a, ogg, flac) and the demo returns
headline measurements, a timed narrative report (report.md), and the
full machine-readable report (report.json) for download.

## The seven layers

- signal: duration, levels, clipping, spectral summary, stereo width.
- rhythm: onsets, tempo, beat grid.
- structure: section boundaries, from the novelty analysis first
  built for Reactor.
- harmony: pitch track, key estimate, per-segment chords with
  confidence margins and honest nulls where evidence is weak.
- words: transcription when a local Whisper engine is installed. In
  this demo none is bundled, so the layer reports not_installed and a
  null transcript. It never fabricates words.
- events: non-music sound segmentation (silence, speech-like,
  music-like, noise-like), a classical reference implementation. The
  trained-model adapter is an interface only, labeled as such.
- expression: the feel, translated. Dynamics contour, groove deviation
  and swing in milliseconds, timbre proxies (brightness, warmth,
  roughness), and a harmonic tension curve.

## The standard

The standard is to hear enough to use the data. A layer that cannot
run reports its real status instead of inventing output, and qualities
not yet translated are listed as open methods, never as impossibilities.

Open methods, researched next: modal key vocabularies beyond major and
minor, trained sound-event models behind the adapter interface, a
polyphonic transcription adapter, emotion naming, and speaker
separation.

## About this payload

The ears package in this folder (./ears) is a vendored snapshot of
fif-ears v0.1.0, so the demo is fully self-contained. Source of truth
and full documentation live in the GitHub repo:
https://github.com/Frozen-In-Flames/fif-ears

This payload is the future Hugging Face Space for fif-ears. It deploys
in the quarterly Hugging Face port batch; until that batch lands, it
is committed in the GitHub repo so anyone can run the demo from source
today.

## Run it yourself

With Docker:

    docker build -t fif-ears-demo .
    docker run -p 7860:7860 fif-ears-demo

Then open http://localhost:7860.

Without Docker:

    python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt
    .venv/bin/python app.py

You also need ffmpeg on the machine: it is the decode path for every
accepted format.

## License

Apache-2.0. Copyright 2026 Aries Hilton / Frozen In Flames.
