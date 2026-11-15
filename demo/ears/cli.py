"""Command line interface.

Usage:
    ears analyze <audio-file> --out <dir>
    python -m ears analyze <audio-file> --out <dir>

Writes report.json and report.md into the output directory and prints
a short headline summary. The contract is deliberately universal: any
agent or tool that can run a command and read two files can use this.
"""

from __future__ import annotations

import argparse
import os
import sys

from .report import analyze_file


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="ears",
        description="Translate an audio file into a structured report "
                    "(report.json) and an AI-readable narrative (report.md).")
    sub = parser.add_subparsers(dest="command", required=True)
    p_an = sub.add_parser("analyze", help="analyze one audio file")
    p_an.add_argument("file", help="path to the audio file")
    p_an.add_argument("--out", default=None,
                      help="output directory (default: <file-stem>-ears)")
    args = parser.parse_args(argv)

    if args.command == "analyze":
        out_dir = args.out or os.path.splitext(
            os.path.basename(args.file))[0] + "-ears"
        report = analyze_file(args.file, out_dir)
        s = report["summary"]
        print(f"file: {report['input']['file']}")
        print(f"duration: {s['duration_s']} s")
        print(f"tempo: {s['tempo_bpm']} BPM")
        print(f"key: {s['key']}")
        print(f"sections: {s['sections']}")
        print(f"words layer: {s['words_status']}")
        print(f"wrote {os.path.join(out_dir, 'report.json')}")
        print(f"wrote {os.path.join(out_dir, 'report.md')}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
