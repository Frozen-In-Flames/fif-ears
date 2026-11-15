# report.json schema

`ears analyze <file> --out <dir>` writes `report.json` with this
shape. Integrators can rely on the field names and types below. Every
layer object carries a `status` string. A layer that cannot run
reports its real status and nulls its result fields; consumers must
check `status` before trusting values.

## Top level

| Field | Type | Meaning |
|---|---|---|
| tool | object | `{name: "fif-ears", version: string}` |
| input | object | `{file: string}` base name of the analyzed file |
| summary | object | headline numbers, see below |
| layers | object | one object per layer, keyed by layer name |

## summary

| Field | Type | Meaning |
|---|---|---|
| duration_s | number | file duration in seconds |
| tempo_bpm | number or null | chosen tempo; null when no rhythmic content |
| key | string or null | e.g. `"D minor"`; null when no pitched content |
| sections | integer | number of structure segments |
| words_status | string | status of the words layer |

## layers.signal (status: "ok")

| Field | Type | Meaning |
|---|---|---|
| duration_s, sample_rate, channels | number, integer, integer | file facts |
| peak_dbfs, rms_dbfs | number | levels in dBFS (silence floors at -80) |
| clipping_samples | integer | samples at or above 0.999 full scale |
| spectral_centroid_hz, spectral_bandwidth_hz, spectral_rolloff_hz | `{mean, std}` | energy-weighted spectral summaries in Hz |
| stereo_width | number | RMS of side signal over RMS of mid signal; 0 for mono |

## layers.rhythm (status: "ok")

| Field | Type | Meaning |
|---|---|---|
| onsets | number[] | attack start times in seconds, refined in the time domain |
| tempo_bpm | number or null | tempo after grid refinement |
| tempo_confidence | number | autocorrelation strength at the chosen lag, 0 to 1 |
| tempo_candidates | `{bpm, strength}[]` | top metrical levels considered, in selection-score order (autocorrelation strength times tempo prior); `strength` is the raw autocorrelation value; includes levels not chosen |
| beat_times | number[] | beat grid times in seconds, snapped to onsets where present |
| beat_period_s, grid_phase_s | number or null | grid period and phase in seconds |

## layers.structure (status: "ok")

| Field | Type | Meaning |
|---|---|---|
| boundaries | number[] | section start times; always begins with 0.0 |
| segments | `{start, end}[]` | sections in seconds |
| novelty | `{t, v}[]` | downsampled smoothed novelty curve (at most about 200 points) |
| note | string, optional | present for files under 8 seconds (single section) |

## layers.harmony (status: "ok")

| Field | Type | Meaning |
|---|---|---|
| pitch_track | `{t, freq, midi, note, confidence}[]` | voiced frames only; freq in Hz, midi as float |
| voiced_fraction | number | share of frames with a detected pitch, 0 to 1 |
| dominant_pitch | `{freq, midi, note}` or null | median voiced pitch |
| key, key_root, key_mode | string or null | Krumhansl-Schmuckler estimate (major/minor vocabulary) |
| key_confidence | number | profile correlation of the winning key |
| mean_chroma | number[12] | mean chroma, C first, sums to about 1 |
| segments | object[] | per structure segment: `{start, end, chord, chord_root, chord_mode, chord_similarity, chord_confidence}`; `chord` is null when the match does not clear the similarity and margin thresholds; `chord_confidence` is the margin over the runner-up template |

## layers.words

| Field | Type | Meaning |
|---|---|---|
| status | string | `"ok"`, `"not_installed"`, or `"error"` |
| engine | string or null | `"faster-whisper"`, `"openai-whisper"`, or null |
| optional_dependencies | string[] | packages that enable this layer: `["openai-whisper", "faster-whisper"]` |
| transcript | string or null | full transcript when status is `"ok"`, else null; never fabricated |
| segments | `{start, end, text}[]` | timed transcript segments |
| note | string, optional | explanation for non-ok statuses |

## layers.events (status: "ok")

| Field | Type | Meaning |
|---|---|---|
| method | string | how labels were produced (classical feature rules) |
| segments | `{start, end, label, confidence}[]` | label is one of `silence`, `speech-like`, `music-like`, `noise-like`; confidence is the share of frames in the segment agreeing with the label |

## layers.expression (status: "ok")

| Field | Type | Meaning |
|---|---|---|
| dynamics.contour | `{t, db}[]` | loudness over time, 100 ms windows, 50 ms hop, downsampled |
| dynamics.global_slope_db_per_s | number | overall loudness slope; positive is crescendo |
| dynamics.dynamic_range_db | number | 95th minus 10th percentile of audible window levels |
| dynamics.per_section | `{start, end, mean_db, slope_db_per_s, dynamic_range_db}[]` | per structure segment |
| groove.beat_deviation_mean_ms | number or null | mean signed attack offset from the beat grid; positive is behind the grid |
| groove.beat_deviation_std_ms | number or null | spread of attack offsets |
| groove.offbeat_delay_ms | number or null | mean delay of eighth-note offbeat onsets; the swing measurement |
| groove.onsets_measured | integer | onsets used |
| timbre.overall, timbre.per_section | object | `{brightness_hz, warmth, roughness}`; warmth is the sub-250 Hz energy share, roughness is normalized spectral flux; per_section adds `{start, end}` |
| tension_curve | `{start, end, chord, tension}[]` | tension 0 to 1, or null when the segment has no reported chord |

Null groove fields mean the rhythm layer found no usable grid. Null
tension means no chord was reported for that segment.
