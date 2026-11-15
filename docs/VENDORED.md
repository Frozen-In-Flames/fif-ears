# Vendored package note

The Python files under `docs/ears/` are byte-identical copies of the
fif-ears package, version 0.1.0, as shipped in this repository under
`ears/`. They are vendored here because GitHub Pages serves only the
`docs/` tree, and the live demo in `docs/index.html` loads these files
into the browser's Python runtime (Pyodide) so the demo runs the real,
unmodified package.

No file was edited for the browser. The only adaptation lives in
`docs/index.html`: the browser decodes audio with the Web Audio API and
hands the samples to the package, standing in for the ffmpeg decode
step in `ears/audio.py`. Every analysis layer runs exactly as shipped.

If the package is updated, re-copy the `.py` files from `ears/` into
`docs/ears/` so the demo never drifts from the shipped code.
