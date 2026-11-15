"""fif-ears: audio perception for agents.

Takes an audio file in, returns a layered, machine-readable report
(report.json) and an AI-readable narrative (report.md) describing what
is in the audio: signal, rhythm, pitch and harmony, structure, words,
sound events, and expression (dynamics, groove, timbre, tension).

Every layer is classical, deterministic signal processing unless its
status field says otherwise. A layer that cannot run reports its real
status instead of inventing output.
"""

__version__ = "0.1.0"

from .report import analyze_file, build_report  # noqa: E402,F401
