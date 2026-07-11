# 3. Development & Operational Runbook

**CircuitKit v1.0.0** — local development setup, Docker-based testing, enterprise CI/CD pipeline, and secure deployment.

## Enterprise Requirements

| Requirement | Specification |
|---|---|
| **Supply chain security** | All dependencies pinned via `requirements.txt` SHA digests; wheels built inside isolated CI containers; no hardcoded tokens |
| **GPU testing matrix** | CI verifies against CUDA 12.x, CPU-only fallback, and MPS (Apple Silicon) |
| **Sign-off gates** | Lint → Type-check → Unit tests → Integration tests → Release sign-off |
| **Audit trail** | Every published wheel has a signed provenance attestation (PyPI trusted publishing) |

## 3.1 Local Developer Environment

### Minimal Setup (no container)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev,benchmarks]"
```

### Editable Docker Setup (hot-reload)

```yaml
# docker-compose.dev.yml
version: "3.9"
services:
  circuitkit:
    build:
      context: .
      dockerfile: Dockerfile.dev
    volumes:
      - .:/app
      - ~/.cache/huggingface:/root/.cache/huggingface
      - ~/.cache/torch:/root/.cache/torch
    command: >
      sh -c "pip install -e /app && circuitkit discover-smart
             --model gpt2 --algorithm eap-ig --task ioi --check-memory"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

```dockerfile
# Dockerfile.dev
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
RUN pip install --upgrade pip
WORKDIR /app
```

### Running Tests Locally

```bash
# CPU-only (fast, mock GPU ops)
pytest tests/ -x -v --ignore=tests/integration

# GPU integration (requires CUDA)
pytest tests/integration/ -x -v

# With coverage
pytest --cov=circuitkit --cov-report=term-missing
```

## 3.2 Enterprise CI/CD Pipeline

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ Code Commit  │──▶│ Lint & Types  │──▶│ GPU Unit     │──▶│ Release      │
│ Push / PR    │   │ ruff + mypy  │   │ Tests        │   │ Sign-Off     │
└──────────────┘   └──────────────┘   └──────────────┘   └──────┬───────┘
                                                                ▼
                                                     ┌──────────────────┐
                                                     │ Build Wheels in  │
                                                     │ Isolated Runner  │
                                                     └────────┬─────────┘
                                                              ▼
                                                     ┌──────────────────┐
                                                     │ Publish to PyPI  │
                                                     │ Trusted Publisher │
                                                     └──────────────────┘
```

### Stage 1 — Lint & Type Check (CPU, <3 min)

```yaml
# .github/workflows/ci.yml (excerpt)
lint:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with: { python-version: "3.11" }
    - run: pip install ruff mypy
    - run: ruff check src/circuitkit/
    - run: mypy src/circuitkit/ --strict
```

### Stage 2 — Unit Tests (CPU, <10 min)

```yaml
unit-tests:
  runs-on: ubuntu-latest
  strategy:
    matrix:
      python-version: ["3.10", "3.11", "3.12"]
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with: { python-version: "${{ matrix.python-version }}" }
    - run: pip install -e ".[dev,benchmarks]"
    - run: pytest tests/ -x -v --ignore=tests/integration
```

### Stage 3 — GPU Integration (CUDA runner, <30 min)

```yaml
gpu-tests:
  runs-on: [self-hosted, gpu]
  steps:
    - uses: actions/checkout@v4
    - run: pip install -e ".[dev,benchmarks]"
    - run: nvidia-smi
    - run: pytest tests/integration/ -x -v
```

### Stage 4 — Build & Release (trusted publishing)

```yaml
release:
  needs: [lint, unit-tests, gpu-tests]
  if: startsWith(github.ref, 'refs/tags/v')
  runs-on: ubuntu-latest
  permissions:
    id-token: write  # for PyPI trusted publishing
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with: { python-version: "3.11" }
    - run: pip install build
    - run: python -m build
    - uses: pypa/gh-action-pypi-publish@release/v1
```

## 3.3 Hardware Matrix Strategy

| Test scenario | Runner | Hardware | Mock strategy |
|---|---|---|---|
| Lint + types | GitHub-hosted | CPU | N/A |
| Unit tests | GitHub-hosted | CPU | Mock CUDA with `torch.device("cpu")` |
| Discovery integration | Self-hosted | 1× GPU (A10G / A100) | Real GPU |
| Evaluation integration | Self-hosted | 1× GPU | Real GPU |
| Pruning integration | Self-hosted | 1× GPU | Real GPU |
| Apple Silicon | Mac mini M2 | MPS | Test via `torch.backends.mps.is_available()` |

```python
# tests/conftest.py — local vs CI detection
import os
import torch

def _device():
    if os.environ.get("CI"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")  # default to CPU for local dev
```

## 3.4 Secured Supply Chain

1. **Dependency pinning**: `requirements.txt` with SHA-256 hashes (`pip freeze --all` + `hashin`)
2. **No hardcoded secrets**: All API keys, tokens, and bucket credentials sourced from environment variables or cloud secret managers
3. **Wheel verification**: CI builds wheels inside an isolated container and verifies SHA-256 before upload
4. **PyPI trusted publishing**: OIDC-based authentication — no API tokens stored in GitHub secrets

```bash
# Verify wheel integrity
sha256sum dist/*.whl > dist/SHA256SUMS
```

## 3.5 Disaster Recovery

| Scenario | RTO | RPO | Procedure |
|---|---|---|---|
| GPU worker crash | <60 s | 0 | Orchestrator detects missing heartbeat, respawns pod, re-queues job |
| Object store outage | <120 s | <5 min | Fall back to local disk cache; retry S3/GCS upload on recovery |
| Database (Redis) failure | <30 s | 0 | Redis Sentinel auto-failover to replica |
| Bad release (PyPI) | <15 min | N/A | YANK the version via PyPI admin; pin previous version in dependents |
