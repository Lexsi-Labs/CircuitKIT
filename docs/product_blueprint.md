# 1. Product Blueprint & Systems Architecture — Roadmap / Aspirational

!!! warning "Aspirational roadmap — none of this is implemented"
    Everything below this notice describes a **possible future** deployment architecture (web ingestion layer, task queue, GPU worker pool, auto-scaling, `circuitkit worker` CLI command, a `circuitkit.api:app` FastAPI service). **None of it exists in `circuitkit` today.** CircuitKit v1.0.0 is a Python library and CLI you run directly (`pip install circuitkit`, `circuitkit discover`, `circuitkit evaluate`, ...) — there is no server, no task queue, no auto-scaling, and no `worker` subcommand. This page is excluded from the published documentation site (see `exclude_docs` in `mkdocs.yml`) and is kept only as a record of a possible future direction.

    For the real, current architecture and usage, see [Flat Typed API](api-reference/flat-api.md), [Pipeline](api-reference/pipeline.md), and [CLI Reference](cli/applications.md).

**CircuitKit** is a unified toolkit for mechanistic circuit discovery, faithfulness evaluation, and model intervention in transformer neural networks. The sections below sketch a hypothetical enterprise deployment built on top of CircuitKit — they are not a description of the shipped product.

## Enterprise Requirements

| Requirement | Specification |
|---|---|
| **Compute isolation** | CPU web nodes and GPU discovery/eval workers run in separate pods to prevent DL workloads from blocking the API |
| **Auto-scaling policy** | Horizontal GPU worker pool auto-scales when the job queue exceeds 500 unprocessed tasks for >60 s |
| **Disaster recovery** | All discovery artifacts and evaluation checkpoints persist to object storage (S3/GCS); failed workers are replaced within 120 s |
| **Cost gating** | Configurable max GPU budget per job; jobs exceeding threshold are redirected to CPU fallback pool |

## 1.1 Microservices Infrastructure Layout

```
                         ┌──────────────────────┐
                         │   Client / CI Shell   │
                         └──────────┬───────────┘
                                    │ REST / gRPC
                                    ▼
                    ┌───────────────────────────────┐
                    │   Web Ingestion Layer (CPU)    │
                    │   FastAPI /  Flask             │
                    │   - auth, rate-limit, dispatch │
                    │   - lightweight (~1 vCPU)      │
                    └──────────────┬────────────────┘
                                  │ Redis / RabbitMQ
                                  ▼
          ┌───────────────────────────────────────────────┐
          │        Task Queue (Redis / Apache Kafka)       │
          │  Serialized discovery/eval/intervention jobs   │
          └──────┬──────────────────────┬─────────────────┘
                 │                      │
                 ▼                      ▼
    ┌────────────────────┐   ┌────────────────────┐
    │  GPU Worker Pool   │   │  CPU Fallback Pool │
    │  (AWS G5 / GCP a2) │   │  (High-mem nodes)  │
    │  - discover_circuit│   │  - light evaluation │
    │  - prune / quantize│   │  - caching / i/o   │
    │  - 6-pillar eval   │   │  - result assembly │
    └────────┬───────────┘   └────────┬───────────┘
             │                        │
             └──────────┬─────────────┘
                        ▼
          ┌────────────────────────────┐
          │   Object Store (S3/GCS)    │
          │   - circuit artifacts.pt   │
          │   - eval reports.json      │
          │   - quantized checkpoints  │
          └────────────────────────────┘
```

## 1.2 Enterprise Scale & Hardware Provisioning Rules

| Component | Min Allocation | Scale-Up Trigger | Max Budget |
|---|---|---|---|
| Web API | 1 node (1 vCPU, 2 GB) | CPU > 80 % for 120 s | 4 nodes |
| GPU Workers | 1 warm instance (1× GPU) | Queue depth > 500 for 60 s | 8 instances |
| CPU Fallback | 1 node (8 vCPU, 32 GB) | GPU pool exhausted | 2 nodes |
| Message Queue | 1 Redis node | — | HA cluster |
| Object Store | 1 bucket | — | S3 / GCS standard |

## 1.3 Fail-Safe & Recovery

**GPU OOM**: Redirects to the CPU fallback thread pool executing on high-memory nodes. The job is retried automatically.

**Worker crash**: The orchestrator detects a missing heartbeat (30 s timeout) and respawns the pod. In-flight jobs are re-queued with a delivery-count header.

**Data corruption**: Every artifact carries a SHA-256 checksum. Corrupted downloads trigger automatic re-fetch from the object store.

## 1.4 Developer Quick-Start (Single Node)

The two CLI commands below are real and work today. Everything after them (the
`docker-compose.dev.yml` with a `worker` service and a `uvicorn circuitkit.api:app`
web service) is part of the aspirational architecture above — `circuitkit worker`
and `circuitkit.api:app` do not exist; there is no FastAPI app and no queue consumer
in the current codebase.

```bash
# Local single-node (no queue, no scaling) — this works today
pip install -e ".[dev,benchmarks]"
circuitkit discover-smart --model gpt2 --algorithm eap-ig --task ioi --check-memory
circuitkit evaluate --model gpt2 --artifact results.pt
```

```yaml
# docker-compose.dev.yml — HYPOTHETICAL, matches the roadmap architecture above,
# not anything that ships with circuitkit today.
services:
  redis:
    image: redis:7-alpine
  worker:
    build: .
    command: circuitkit worker --queue redis:6379   # does not exist
    volumes:
      - .:/app
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
  web:
    build: .
    command: uvicorn circuitkit.api:app --host 0.0.0.0 --reload  # does not exist
    ports:
      - "8000:8000"
```

## 1.5 Network & Security

- All inter-service communication over mutual TLS
- Artifact bucket access via ephemeral STS tokens (AWS) / workload identity (GCP)
- GPU workers run in a private subnet with no public IP
- Rate limiting: 100 req / min per API key
