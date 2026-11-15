"""Report assembly: run the layers, write report.json and report.md.

report.md is the multimodal translation written for an AI reader: a
timed narrative combining the layers, in plain factual language with
numbers, so any model can reason over the audio without hearing it.
"""

from __future__ import annotations

import json
import os

from . import __version__
from .audio import load_audio
from .layers import events, expression, harmony, rhythm, signal, structure, words

LAYER_ORDER = ["signal", "rhythm", "structure", "harmony",
               "words", "events", "expression"]


def build_report(path: str) -> dict:
    audio = load_audio(path)
    layers = {}
    layers["signal"] = signal.analyze(audio)
    layers["rhythm"] = rhythm.analyze(audio)
    layers["structure"] = structure.analyze(audio)
    segments = layers["structure"]["segments"]
    layers["harmony"] = harmony.analyze(audio, segments=segments)
    layers["words"] = words.analyze(audio, path=path)
    layers["events"] = events.analyze(audio)
    layers["expression"] = expression.analyze(
        audio, layers["rhythm"], layers["harmony"], segments)

    r = layers["rhythm"]
    h = layers["harmony"]
    summary = {
        "duration_s": layers["signal"]["duration_s"],
        "tempo_bpm": r["tempo_bpm"],
        "key": h["key"],
        "sections": len(segments),
        "words_status": layers["words"]["status"],
    }
    return {
        "tool": {"name": "fif-ears", "version": __version__},
        "input": {"file": os.path.basename(path)},
        "summary": summary,
        "layers": layers,
    }


def _fmt_time(s: float) -> str:
    m, sec = divmod(float(s), 60)
    return f"{int(m)}:{sec:05.2f}"


def _dominant_event(events_layer: dict, start: float, end: float):
    best_label, best_overlap = None, 0.0
    for seg in events_layer["segments"]:
        overlap = min(end, seg["end"]) - max(start, seg["start"])
        if overlap > best_overlap:
            best_overlap, best_label = overlap, seg["label"]
    return best_label


def render_markdown(report: dict) -> str:
    layers = report["layers"]
    sig, rhy, st = layers["signal"], layers["rhythm"], layers["structure"]
    har, wrd, evt, exp = (layers["harmony"], layers["words"],
                          layers["events"], layers["expression"])
    lines = []
    lines.append(f"# fif-ears report: {report['input']['file']}")
    lines.append("")
    lines.append(
        "This report is a machine perception translation of an audio "
        "file: measurements and classifications an AI can reason over. "
        "It is not a recording of anyone's felt experience of the sound. "
        "Qualities not yet translated into measurements are listed as "
        "open methods at the end, never as impossibilities.")
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append(f"- Duration: {sig['duration_s']} s, sample rate "
                 f"{sig['sample_rate']} Hz, {sig['channels']} channel(s)")
    lines.append(f"- Level: peak {sig['peak_dbfs']} dBFS, RMS "
                 f"{sig['rms_dbfs']} dBFS, clipping samples: "
                 f"{sig['clipping_samples']}")
    lines.append(f"- Stereo width (side/mid RMS): {sig['stereo_width']}")
    if rhy["tempo_bpm"] is not None:
        lines.append(f"- Tempo: {rhy['tempo_bpm']} BPM "
                     f"(confidence {rhy['tempo_confidence']}), "
                     f"{len(rhy['beat_times'])} beats on the grid, "
                     f"{len(rhy['onsets'])} onsets detected")
        alts = [c for c in rhy.get("tempo_candidates", [])
                if abs(c["bpm"] - rhy["tempo_bpm"]) > 3.0]
        if alts:
            lines.append("- Other metrical levels measured: "
                         + ", ".join(f"{c['bpm']} BPM (strength "
                                     f"{c['strength']})" for c in alts))
    else:
        lines.append("- Tempo: none detected (no rhythmic content measured)")
    if har["key"] and har["voiced_fraction"] >= 0.2:
        lines.append(f"- Key estimate: {har['key']} "
                     f"(correlation {har['key_confidence']})")
    elif har["key"]:
        lines.append(f"- Key estimate: {har['key']} "
                     f"(correlation {har['key_confidence']}), but only "
                     f"{har['voiced_fraction']} of frames carried a pitch, "
                     f"so treat the key as unreliable for this file")
    if har["dominant_pitch"]:
        dp = har["dominant_pitch"]
        lines.append(f"- Dominant pitch: {dp['note']} at {dp['freq']} Hz, "
                     f"voiced fraction {har['voiced_fraction']}")
    lines.append(f"- Sections: {len(st['segments'])}")
    lines.append("")

    groove = exp["groove"]
    if groove["offbeat_delay_ms"] is not None or \
            groove["beat_deviation_mean_ms"] is not None:
        lines.append("## Groove")
        lines.append("")
        if groove["beat_deviation_mean_ms"] is not None:
            mean_ms = groove["beat_deviation_mean_ms"]
            direction = "behind" if mean_ms > 0 else "ahead of"
            lines.append(
                f"- Attacks land on average {abs(mean_ms)} ms "
                f"{direction} the beat grid, with a spread of "
                f"{groove['beat_deviation_std_ms']} ms")
        if groove["offbeat_delay_ms"] is not None:
            lines.append(
                f"- Offbeat (eighth-note) onsets sit "
                f"{groove['offbeat_delay_ms']} ms from their grid points "
                f"on average: positive means laid back, swung")
        lines.append("")

    lines.append("## Words")
    lines.append("")
    if wrd["status"] == "ok":
        lines.append(f"Engine: {wrd['engine']}")
        lines.append("")
        lines.append(wrd["transcript"] or "(no speech transcribed)")
    else:
        lines.append(f"Status: {wrd['status']}. {wrd.get('note', '')}")
    lines.append("")

    lines.append("## Timed narrative")
    lines.append("")
    dyn_sections = {round(s["start"], 2): s
                    for s in exp["dynamics"]["per_section"]}
    tim_sections = {round(s["start"], 2): s
                    for s in exp["timbre"]["per_section"]}
    ten_sections = {round(s["start"], 2): s
                    for s in exp["tension_curve"]}
    har_sections = {round(s["start"], 2): s for s in har["segments"]}
    for seg in st["segments"]:
        start, end = seg["start"], seg["end"]
        key = round(float(start), 2)
        parts = []
        label = _dominant_event(evt, start, end)
        if label:
            parts.append(f"the sound is mostly {label}")
        hs = har_sections.get(key)
        if hs and hs["chord"]:
            parts.append(f"harmony reads as a {hs['chord']} chord "
                         f"(similarity {hs['chord_similarity']})")
        elif hs:
            parts.append("no chord clears the confidence margin here")
        ds = dyn_sections.get(key)
        if ds:
            slope = ds["slope_db_per_s"]
            if slope > 0.3:
                trend = f"loudness swells at {slope} dB per second"
            elif slope < -0.3:
                trend = f"loudness falls at {slope} dB per second"
            else:
                trend = "loudness holds steady"
            parts.append(f"mean level {ds['mean_db']} dB, {trend}, "
                         f"dynamic range {ds['dynamic_range_db']} dB")
        ts = tim_sections.get(key)
        if ts:
            parts.append(f"brightness centroid {ts['brightness_hz']} Hz, "
                         f"warmth (low-band share) {ts['warmth']}, "
                         f"roughness proxy {ts['roughness']}")
        tn = ten_sections.get(key)
        if tn and tn["tension"] is not None:
            parts.append(f"harmonic tension {tn['tension']} on a 0 to 1 scale")
        lines.append(f"- {_fmt_time(start)} to {_fmt_time(end)}: "
                     + "; ".join(parts) + ".")
    lines.append("")
    lines.append("## Events timeline")
    lines.append("")
    for seg in evt["segments"]:
        lines.append(f"- {_fmt_time(seg['start'])} to {_fmt_time(seg['end'])}: "
                     f"{seg['label']} (confidence {seg['confidence']})")
    lines.append("")
    lines.append("## Open methods")
    lines.append("")
    lines.append("- Trained sound-event classification (YAMNet or PANNs "
                 "style): the adapter interface is defined in the events "
                 "layer; the classical reference implementation runs today.")
    lines.append("- Word-level transcription on machines without a local "
                 "Whisper install: the words layer reports its status "
                 "until the optional dependency is present.")
    lines.append("- Emotion and felt-quality labels: correlates are "
                 "measured in the expression layer (dynamics, groove, "
                 "timbre, tension); naming feelings from them is a "
                 "translation still being researched.")
    lines.append("")
    return "\n".join(lines)


def analyze_file(path: str, out_dir: str) -> dict:
    """Run the full pipeline and write report.json and report.md."""
    report = build_report(path)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    with open(os.path.join(out_dir, "report.md"), "w") as fh:
        fh.write(render_markdown(report))
    return report
