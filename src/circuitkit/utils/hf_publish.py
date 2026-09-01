"""HuggingFace Hub publishing with CircuitKIT / Lexsi Labs model-card branding.

`brand_hf_repo` uploads the packaged logo + README onto any Hub repo after
weights, a circuit artifact, a quantized model, or GGUF files are pushed.
Logos ship inside the package (`circuitkit/assets/*.png`) and are copied into
the destination repo. README image srcs point at that repo — never a personal
Hugging Face CDN upload URL.

Notebooks stay short: they call `circuit.push_to_hub` /
`push_quantized_to_hub` / `push_gguf_to_hub`. Branding is not pasted into cells.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Union

logger = logging.getLogger(__name__)

CIRCUITKIT_REPO_URL = "https://github.com/Lexsi-Labs/circuitkit"
LEXSI_URL = "https://lexsi.ai/"

_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
_LOGO_ASSET = _ASSETS_DIR / "circuitkit_logo.png"
_LOGO_REPO_NAME = "circuitkit_logo.png"

_VALID_KINDS = ("circuit", "model", "quant", "gguf")

GGUF_QUANT_PRESETS = {
    "Q2_K": "q2_k",
    "Q3_K_M": "q3_k_m",
    "Q4_K_S": "q4_k_s",
    "Q4_K_M": "q4_k_m",
    "Q5_K_S": "q5_k_s",
    "Q5_K_M": "q5_k_m",
    "Q6_K": "q6_k",
    "Q8_0": "q8_0",
}

_BNB_PRESETS = {
    "nf4": dict(load_in_4bit=True, bnb_4bit_quant_type="nf4"),
    "fp4": dict(load_in_4bit=True, bnb_4bit_quant_type="fp4"),
    "bf4": dict(load_in_4bit=True, bnb_4bit_quant_type="fp4"),
    "int8": dict(load_in_8bit=True),
}

_CONVERTER_URL = (
    "https://raw.githubusercontent.com/ggerganov/llama.cpp/refs/tags/b3691/"
    "convert_hf_to_gguf.py"
)


def _resolve_token(token: Optional[str] = None) -> str:
    token = (
        token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    if not token:
        raise ValueError(
            "No HuggingFace token found. Pass token=..., set HF_TOKEN, "
            "or run huggingface-cli login."
        )
    return token


def _hub_asset_url(repo_id: str, filename: str) -> str:
    return f"https://huggingface.co/{repo_id}/resolve/main/{filename}"


def _upload_packaged_asset(api, repo_id: str, token: str, local: Path, name: str) -> Optional[str]:
    if not local.exists():
        logger.warning("Packaged branding asset missing: %s", local)
        return None
    api.upload_file(
        path_or_fileobj=str(local),
        path_in_repo=name,
        repo_id=repo_id,
        token=token,
    )
    return _hub_asset_url(repo_id, name)


def _branding_header(logo_url: Optional[str]) -> str:
    if not logo_url:
        return ""
    return f"""<div align="center">
  <table border="0" cellspacing="0" cellpadding="0" style="border: none; border-collapse: collapse;">
    <tr>
      <td align="center" style="border: none; vertical-align: middle;">
        <a href="{LEXSI_URL}"><img src="{logo_url}" alt="CircuitKIT" style="height: 60px; border-radius: 12px;"/></a>
      </td>
    </tr>
  </table>
</div>
"""


def _usage_block(kind: str, repo_id: str, base_model: str, gguf_files: Optional[Iterable[str]]) -> str:
    files = [str(x) for x in (gguf_files or [])]
    if kind == "circuit":
        return f"""```python
from huggingface_hub import snapshot_download
from circuitkit import Circuit

local = snapshot_download("{repo_id}")
circuit = Circuit.from_artifact(f"{{local}}/circuit.pt")
```"""
    if kind == "quant":
        return f"""```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
```

This repo is a BitsAndBytes quantized checkpoint. Keep nf4 / bf4 / int8 in **separate** Hub repos — each save writes its own `config.json` `quantization_config` at the repo root."""
    if kind == "gguf":
        listed = "\n".join(f"- `{f}`" for f in files) if files else "- (GGUF files in this repo)"
        sample = files[0] if files else "model.gguf"
        return f"""Several GGUF files can live in **one** Hub repo (different filenames). That is the usual layout.

{listed}

```python
from llama_cpp import Llama
llm = Llama.from_pretrained(repo_id="{repo_id}", filename="{sample}")
```"""
    return f"""```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
```"""


def brand_hf_repo(
    repo_id: str,
    kind: str = "model",
    base_model: str = "",
    algorithm: str = "",
    private: bool = False,
    token: Optional[str] = None,
    extra_notes: str = "",
    gguf_files: Optional[Iterable[str]] = None,
) -> str:
    """Create the repo if needed, upload the packaged CircuitKIT logo, write README.md.

    Does not upload weights. Call after circuit / checkpoint / quant / GGUF push.
    kind: circuit | model | quant | gguf
    """
    from huggingface_hub import HfApi

    kind = (kind or "model").lower()
    if kind not in _VALID_KINDS:
        raise ValueError(f"kind must be one of {_VALID_KINDS}, got {kind!r}")

    token = _resolve_token(token)
    api = HfApi(token=token)
    built_on = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    api.create_repo(repo_id, private=private, exist_ok=True, token=token)

    logo_url = _upload_packaged_asset(api, repo_id, token, _LOGO_ASSET, _LOGO_REPO_NAME)
    header = _branding_header(logo_url)

    model_name = repo_id.split("/")[-1]
    tag = (algorithm or "circuitkit").lower().replace(" ", "-").replace("(", "").replace(")", "")
    usage = _usage_block(kind, repo_id, base_model, gguf_files)
    if base_model:
        base_row = f"| **Model** | `{base_model}` |"
    else:
        base_row = "| **Model** | — |"

    readme = f"""---
library_name: {"gguf" if kind == "gguf" else "transformers"}
tags:
  - circuitkit
  - {tag}
  - {kind}
---

{header}
# {model_name}

Built using [CircuitKIT]({CIRCUITKIT_REPO_URL}).

| | |
|---|---|
{base_row}
| **Algorithm** | {algorithm or "—"} |
| **Artifact** | {kind} |
| **Published** | {built_on} |

{extra_notes}

## Usage

{usage}
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(readme)
        readme_path = f.name
    try:
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            token=token,
        )
    finally:
        os.remove(readme_path)

    url = f"https://huggingface.co/{repo_id}"
    logger.info("Branded %s", url)
    return url


def _brand_safe(repo_id: str, kind: str = "model", token: Optional[str] = None, **kw) -> str:
    """Brand the repo; a branding failure does not undo a successful push."""
    try:
        return brand_hf_repo(repo_id, kind=kind, token=token, **kw)
    except Exception as e:
        logger.warning("Model card branding skipped for %s: %s", repo_id, e)
        return f"https://huggingface.co/{repo_id}"


def _local_model_dir(path: Union[str, Path]) -> Path:
    path = Path(path)
    if path.exists():
        return path / "model" if (path / "model").is_dir() else path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(str(path)))


def push_folder_to_hub(
    folder: Union[str, Path],
    repo_id: str,
    private: bool = False,
    token: Optional[str] = None,
    repo_type: str = "model",
    kind: str = "circuit",
    base_model: str = "",
    algorithm: str = "",
) -> str:
    """Upload a local directory to a Hub repo, then brand the model card."""
    from huggingface_hub import HfApi

    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"folder does not exist: {folder}")
    token = _resolve_token(token)
    api = HfApi(token=token)
    api.create_repo(
        repo_id, private=private, exist_ok=True, token=token, repo_type=repo_type
    )
    api.upload_folder(
        folder_path=str(folder), repo_id=repo_id, repo_type=repo_type, token=token
    )
    _brand_safe(
        repo_id,
        kind=kind,
        private=private,
        token=token,
        base_model=base_model,
        algorithm=algorithm,
    )
    return f"https://huggingface.co/{repo_id}"


def push_quantized_path_to_hub(
    path,
    repo_id,
    quantization="nf4",
    private=False,
    token=None,
    tokenizer_path=None,
    **kw,
):
    """Load ``path`` with BitsAndBytes and push. Same presets as AlignTune."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    token = _resolve_token(token)
    key = str(quantization).lower()
    if key not in _BNB_PRESETS:
        raise ValueError(f"quantization must be one of {sorted(_BNB_PRESETS)}, got {quantization!r}")
    preset = dict(_BNB_PRESETS[key])
    if "bnb_4bit_quant_type" in preset:
        preset["bnb_4bit_compute_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        path, quantization_config=BitsAndBytesConfig(**preset), device_map="auto"
    )
    tok = AutoTokenizer.from_pretrained(tokenizer_path or path)
    try:
        model.push_to_hub(repo_id, token=token, private=private)
        tok.push_to_hub(repo_id, token=token, private=private)
    finally:
        del model
    _brand_safe(
        repo_id,
        kind="quant",
        private=private,
        token=token,
        base_model=str(path),
        extra_notes=f"BitsAndBytes `{key}` quantization.",
        **{k: v for k, v in kw.items() if k in {"algorithm"}},
    )
    return f"https://huggingface.co/{repo_id}"


class GGUFExporter:
    """llama.cpp HF→GGUF converter. Same presets as AlignTune ``GGUFExporter``."""

    def __init__(self, output_dir=None, quantization="Q5_K_M", converter=None):
        self.output_dir = output_dir
        self.quantization = quantization
        self.converter = converter

    def export(self, checkpoint_path):
        return export_gguf(
            checkpoint_path,
            quantization=self.quantization or "Q5_K_M",
            output_dir=self.output_dir,
        )


def export_gguf(path, quantization="Q5_K_M", output_dir=None) -> Path:
    """Convert a HF checkpoint or Hub id to a GGUF file (llama.cpp converter)."""
    quant = str(quantization).upper()
    if quant not in GGUF_QUANT_PRESETS:
        raise ValueError(f"Unknown GGUF quantization {quantization!r}. Valid: {list(GGUF_QUANT_PRESETS)}")
    model_dir = _local_model_dir(path)
    out_dir = Path(output_dir or f"./gguf_{quant}")
    out_dir.mkdir(parents=True, exist_ok=True)
    gguf_path = out_dir / "model.gguf"

    with tempfile.TemporaryDirectory() as tmp:
        converter = Path(tmp) / "convert_hf_to_gguf.py"
        urllib.request.urlretrieve(_CONVERTER_URL, converter)
        cmd = [sys.executable, str(converter), str(model_dir), "--outfile", str(gguf_path)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"GGUF conversion failed: {proc.stderr or proc.stdout}")

    quantize_exe = shutil.which("llama-quantize")
    if quantize_exe:
        qpath = out_dir / f"model-{quant.lower()}.gguf"
        qproc = subprocess.run(
            [quantize_exe, str(gguf_path), str(qpath), GGUF_QUANT_PRESETS[quant]],
            capture_output=True,
            text=True,
        )
        if qproc.returncode == 0:
            return qpath
        logger.warning("llama-quantize failed; uploading unquantized GGUF")
    return gguf_path


def push_gguf_path_to_hub(
    path, repo_id, quantization="Q5_K_M", private=False, token=None, **kw
):
    """Export GGUF then upload as ``model-<quant>.gguf``. Same names as AlignTune."""
    from huggingface_hub import HfApi

    token = _resolve_token(token)
    quant = str(quantization).upper()
    out = GGUFExporter(output_dir=f"./gguf_{quant}", quantization=quant).export(path)
    api = HfApi(token=token)
    api.create_repo(repo_id, private=private, exist_ok=True, token=token)
    filename = f"model-{quant.lower()}.gguf"
    api.upload_file(
        path_or_fileobj=str(out), path_in_repo=filename, repo_id=repo_id, token=token
    )
    _brand_safe(
        repo_id,
        kind="gguf",
        private=private,
        token=token,
        base_model=str(path),
        gguf_files=[filename],
        **{k: v for k, v in kw.items() if k in {"algorithm"}},
    )
    return f"https://huggingface.co/{repo_id}"
