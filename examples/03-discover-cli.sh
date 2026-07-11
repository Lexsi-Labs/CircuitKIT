#!/usr/bin/env bash
# 02 - Circuit discovery via the CLI.
#
# The `circuitkit` console script exposes the same discovery pipeline as the
# Python API. Run this from the repository root.
#
# Run:
#     bash examples/02_discover_cli.sh

set -euo pipefail

mkdir -p ./results

# `circuitkit discover` runs discovery on a built-in task.
# --algorithm accepts any of the discovery algorithms (eap-ig is the default).
circuitkit discover \
    --model gpt2 \
    --algorithm eap-ig \
    --task ioi \
    --level node \
    --scope heads \
    --sparsity 0.2 \
    --batch-size 2 \
    --num-examples 16 \
    --ig-steps 2 \
    --output ./results/example_circuit_cli.pt

echo
echo "Discovery finished -> ./results/example_circuit_cli.pt"
echo
echo "Other useful commands:"
echo "  circuitkit --help                 # all sub-commands"
echo "  circuitkit list-models            # supported models"
echo "  circuitkit discover-smart ...     # memory-aware discovery"
echo "  circuitkit evaluate --model gpt2 --artifact ./results/example_circuit_cli.pt"
