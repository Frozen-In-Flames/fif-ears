"""fif-ears live demo.

Upload an audio file and the real fif-ears package (vendored in ./ears,
a snapshot of v0.1.0) analyzes it: headline measurements, the full
report.md narrative rendered as markdown, and downloads for report.json
and report.md.

Errors are reported plainly with the real error message. A layer that
cannot run reports its real status instead of inventing output.
"""

from __future__ import annotations

import os
import tempfile

import gradio as gr

from ears.report import analyze_file, render_markdown

SUPPORTED = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}

INTRO = """# fif-ears: audio perception for agents

Upload an audio file (wav, mp3, m4a, ogg, flac) and fif-ears translates
it into measurements any human or model can reason over, across seven
layers: signal, rhythm, structure, harmony, words, events, and
expression (dynamics, groove, timbre, tension).

The standard is to hear enough to use the data. Nothing here is faked:
a layer that cannot run says so. One honest note for this demo: the
words layer needs a local Whisper transcription engine, and none is
bundled here, so it will report not_installed while every other layer
runs for real.
"""


def _fmt(value, unit=""):
    if value is None:
        return "none detected"
    return f"{value}{unit}"


def _headline(report: dict) -> str:
    summary = report["summary"]
    layers = report["layers"]
    har = layers["harmony"]
    exp = layers["expression"]
    dp = har.get("dominant_pitch")
    pitch = f"{dp['note']} at {dp['freq']} Hz" if dp else "none detected"
    groove = exp["groove"]
    gd = groove.get("beat_deviation_mean_ms")
    if gd is not None:
        groove_txt = (f"{gd} ms mean deviation from the beat grid, "
                      f"spread {groove.get('beat_deviation_std_ms')} ms")
    else:
        groove_txt = "none detected"
    lines = [
        "## Headline results",
        "",
        f"- Duration: {summary['duration_s']} s",
        f"- Tempo: {_fmt(summary['tempo_bpm'], ' BPM')}",
        f"- Key: {_fmt(summary['key'])}",
        f"- Dominant pitch: {pitch}",
        f"- Sections: {summary['sections']}",
        f"- Dynamic range: {exp['dynamics']['dynamic_range_db']} dB",
        f"- Groove: {groove_txt}",
        f"- Words layer status: {summary['words_status']}",
    ]
    return "\n".join(lines)


def analyze(uploaded):
    """Run fif-ears on one uploaded file. Returns headline markdown,
    the report.md narrative, and the two report file paths."""
    if uploaded is None:
        return ("## No file yet\n\nUpload an audio file first.",
                "", None, None)
    path = uploaded if isinstance(uploaded, str) else uploaded.name
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED:
        return (f"## Unsupported file\n\n{os.path.basename(path)} has "
                f"extension {ext or '(none)'}. Supported formats: "
                "wav, mp3, m4a, ogg, flac.",
                "", None, None)
    out_dir = tempfile.mkdtemp(prefix="fif-ears-demo-")
    try:
        report = analyze_file(path, out_dir)
    except Exception as exc:  # plain error, real message, no fake result
        return (f"## Analysis failed\n\n{type(exc).__name__}: {exc}",
                "", None, None)
    return (
        _headline(report),
        render_markdown(report),
        os.path.join(out_dir, "report.json"),
        os.path.join(out_dir, "report.md"),
    )


with gr.Blocks(title="fif-ears") as demo:
    gr.Markdown(INTRO)
    upload = gr.File(
        label="Audio file",
        file_types=[".wav", ".mp3", ".m4a", ".ogg", ".flac"],
    )
    run = gr.Button("Analyze", variant="primary")
    headline = gr.Markdown()
    narrative = gr.Markdown()
    dl_json = gr.File(label="Download report.json")
    dl_md = gr.File(label="Download report.md")
    run.click(analyze, inputs=upload,
              outputs=[headline, narrative, dl_json, dl_md])


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
