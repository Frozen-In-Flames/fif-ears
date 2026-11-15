# Evaluation datasets

Free academic datasets suitable for calibrating and evaluating the
fif-ears layers. Documented as pointers only: this repo downloads
nothing, bundles no third-party data, and every dataset stays under
its own license at its source. Use them to build evaluation harnesses
against the layer contracts in SCHEMA.md.

## Words layer: LibriSpeech

- What: about 1,000 hours of read English speech with transcripts,
  derived from LibriVox audiobooks, segmented and aligned for speech
  recognition evaluation (standard train/dev/test splits, including
  the widely used test-clean and test-other sets).
- Source: OpenSLR (resource SLR12), from the LibriSpeech paper by
  Panayotov, Chen, Povey, and Khudanpur (2015).
- License: CC BY 4.0.
- Calibrates: transcript accuracy (word error rate) for the words
  layer when a Whisper engine is installed.

## Events layer: FSD50K

- What: about 51,000 human-labeled audio clips (over 100 hours) across
  200 AudioSet ontology classes: speech, music, everyday sounds,
  nature, machines.
- Source: Freesound, curated by the Music Technology Group at
  Universitat Pompeu Fabra, Barcelona (Fonseca et al.), distributed
  via Zenodo.
- License: the dataset compilation is CC BY 4.0; individual clips keep
  their Freesound licenses (CC0, CC BY, or CC BY-NC), recorded per
  clip in the metadata.
- Calibrates: event segment labels and boundaries for the events
  layer, per class.

## Music layers: MUSDB18

- What: 150 full-length music tracks with isolated stems (vocals,
  drums, bass, other) and the mixture, the standard corpus for music
  source separation and a clean ground truth for structure and harmony
  work on real mixes.
- Source: assembled by Rafii et al. (2017); distributed via Zenodo,
  with tracks sourced from the MedleyDB and DSD100 corpora and native
  productions.
- License: CC BY-NC-SA 4.0.
- Calibrates: rhythm, harmony, structure, and expression layers on
  real produced music with known stems (for example, run the layers
  on the drums stem alone to test tempo and groove in isolation).

## Pitch layer: MAESTRO

- What: about 200 hours of virtuoso piano performances (International
  Piano-e-Competition), audio aligned to precise MIDI note onsets,
  velocities, and pedal data.
- Source: Google (Hawthorne et al., 2019), distributed from the
  Magenta project site.
- License: CC BY-NC-SA 4.0.
- Calibrates: pitch track accuracy and note-level timing for the
  harmony layer on dense polyphonic piano.

## Ground rules

- Download datasets only from the sources above, under their
  licenses, into space outside this repo.
- Evaluation code may reference dataset paths by configuration; it
  must never copy dataset audio into this repo.
- Report evaluation numbers with the dataset split named (for
  example, LibriSpeech test-clean), so results stay reproducible.
