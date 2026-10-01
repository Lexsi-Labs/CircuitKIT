<!-- circuitkit-logo -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="../../docs/assets/circuitkit-logo-white.png">
    <img src="../../docs/assets/circuitkit-logo-black.png" width="200" alt="CircuitKIT">
  </picture>
</p>

# Visualization examples

- **`graph-viz-example.py`** — end-to-end script showing how to build a
  `CircuitScores` object, render it with `CircuitGraphVisualizer`, apply
  threshold filtering, and export an interactive HTML graph.

- **`ioi_eap-ig.html`** — a rendered sample of that export: the IOI
  (Indirect Object Identification) circuit for GPT-2 discovered with EAP-IG.
  Open it in any browser to explore the interactive graph — hover nodes for
  layer/head/score, drag to pan, and scroll to zoom. It is a self-contained
  static file (the d3 and elkjs layout libraries load from a CDN, so an
  internet connection is needed the first time you open it).

This same visualization is embedded live in the docs under
[User Guide → Visualization](../../docs/user-guide/visualization.md).
