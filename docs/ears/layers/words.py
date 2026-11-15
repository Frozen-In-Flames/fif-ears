"""Layer 5, words: speech transcription when a local engine exists.

Checks for a locally installed Whisper implementation (the openai
whisper package, the faster-whisper package, or a whisper executable).
If none is installed, the layer reports status "not_installed" and the
exact optional dependency names, with a null transcript. It never
fabricates a transcript.
"""

from __future__ import annotations

import importlib.util
import shutil

from ..audio import AudioData

OPTIONAL_DEPENDENCIES = ["openai-whisper", "faster-whisper"]


def _available_engine():
    if importlib.util.find_spec("faster_whisper") is not None:
        return "faster-whisper"
    if importlib.util.find_spec("whisper") is not None:
        return "openai-whisper"
    if shutil.which("whisper") is not None:
        return "whisper-cli"
    return None


def analyze(audio: AudioData, path: str | None = None) -> dict:
    engine = _available_engine()
    if engine is None:
        return {
            "status": "not_installed",
            "engine": None,
            "optional_dependencies": OPTIONAL_DEPENDENCIES,
            "transcript": None,
            "segments": [],
            "note": ("no local Whisper implementation is installed; "
                     "install one of the optional dependencies to enable "
                     "this layer"),
        }
    # An engine is installed: transcribe with its smallest default model.
    # This path is exercised only on machines that have the dependency.
    try:
        if engine == "faster-whisper":
            from faster_whisper import WhisperModel
            model = WhisperModel("tiny")
            segments, _info = model.transcribe(path)
            segs = [{"start": round(float(s.start), 3),
                     "end": round(float(s.end), 3),
                     "text": s.text.strip()} for s in segments]
        elif engine == "openai-whisper":
            import whisper
            model = whisper.load_model("tiny")
            result = model.transcribe(path)
            segs = [{"start": round(float(s["start"]), 3),
                     "end": round(float(s["end"]), 3),
                     "text": s["text"].strip()} for s in result["segments"]]
        else:
            return {
                "status": "error",
                "engine": engine,
                "optional_dependencies": OPTIONAL_DEPENDENCIES,
                "transcript": None,
                "segments": [],
                "note": "whisper executable found but no supported Python API",
            }
    except Exception as exc:  # report honestly, never invent text
        return {
            "status": "error",
            "engine": engine,
            "optional_dependencies": OPTIONAL_DEPENDENCIES,
            "transcript": None,
            "segments": [],
            "note": f"transcription failed: {exc}",
        }
    return {
        "status": "ok",
        "engine": engine,
        "optional_dependencies": OPTIONAL_DEPENDENCIES,
        "transcript": " ".join(s["text"] for s in segs),
        "segments": segs,
    }
