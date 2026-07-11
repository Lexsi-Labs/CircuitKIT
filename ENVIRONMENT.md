# Environment & Reproducibility

The CircuitKit experiments were run and verified in a fixed environment. To
reproduce the results (or run a shard on another machine), match it exactly.

## Base container

Everything runs inside the **NVIDIA NeMo Framework container, v25.09**:

```
nvcr.io/nvidia/nemo:25.09
```

| Provided by the image | Version |
|---|---|
| OS | Ubuntu 24.04.3 LTS |
| Python | 3.12 |
| CUDA | 13.0.1 |
| PyTorch | 2.9.0a0+50eac81 (NVIDIA build) |
| vLLM | 0.10.1.1 (NVIDIA build 25.09) |
| flash-attn / apex | container builds |

`torch`, `vllm`, `flash-attn`, `apex` are **NVIDIA container builds** — they
are *not* on PyPI and must come from this image. `requirements.txt` therefore
does **not** list them, and a plain `pip install` of `requirements-lock.txt`
on a non-NeMo machine will fail on those `file://` wheels. The container *is*
the pin.

## Setup (inside the container)

```bash
git clone <repo> circuitkit && cd circuitkit
pip install -r requirements.txt     # on-top deps (transformers, quanto, …)
pip install -e .                    # circuitkit itself, editable
huggingface-cli login               # token with Lexsi org write (checkpoint push)
```

## Verify the environment

```bash
python - <<'PY'
import importlib.metadata as md
expect = {
    "torch": "2.9.0a0", "vllm": "0.10.1.1", "transformers": "4.57.6",
    "optimum-quanto": "0.2.7", "llmcompressor": "0.10.0.2",
    "lm_eval": "0.4.8", "transformer-lens": "2.18.0", "datasets": "4.6.0",
}
for p, want in expect.items():
    got = md.version(p)
    print(f"  {p:18s} {got:24s} {'OK' if got.startswith(want) else 'MISMATCH (want '+want+')'}")
PY
```

## Pinned manifests

| File | What it is |
|---|---|
| `requirements.txt` | pip deps installed on top of the NeMo image (pinned to the verified versions) |
| `requirements-lock.txt` | full `pip freeze` of the verified env (644 pkgs) — exact audit/diff reference; not a standalone install recipe (container `file://` wheels) |
| `pyproject.toml` | the `circuitkit` package's own dependency spec (library lower-bounds) |

## GPU

Verified on a single **NVIDIA H200 (140 GB)**, driver 550.127.08. The grid
runs on one H200; `launch_concurrent.sh` runs 3 shards per H200 concurrently.
Smaller GPUs need lower `batch_size` / `gpu_memory_utilization` in the configs.
