"""Reproducible environment preflight for the multi-model integration
(SmolLM3-3B, Aya Expanse 8B, Command R7B — see the integration guide's Stage 0).

Codifies the checks the plan asks to run by hand so the environment contract
is re-runnable rather than a one-off: TransformerLens 2.18.x or 3.8.x (the
``_tl_compat`` port's version guard target), transformers new enough to import
all three HF architectures used by this port (``cohere``, ``cohere2``,
``smollm3``), and the support libraries the ports/tests rely on.

The CUDA/GPU checks are informational assertions guarded by
``torch.cuda.is_available()`` so this module stays green (skipping, not
failing) on CPU-only boxes -- the real-weight Group F gates are what actually
require a GPU, not this preflight.

Run directly with ``make check-env`` (``pytest tests/test_environment.py -v``).
"""

from __future__ import annotations

import importlib.metadata

import pytest
from packaging.version import parse as _parse_version

pytestmark = pytest.mark.integration


class TestGpuAndTorch:
    def test_torch_cuda_available_when_gpu_present(self):
        import torch

        if not torch.cuda.is_available():
            pytest.skip("no CUDA device visible in this environment")
        assert torch.cuda.device_count() >= 1
        name = torch.cuda.get_device_name(0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        assert total_gb > 0, f"reported 0 GB VRAM on device {name!r}"


class TestTransformerLensVersion:
    def test_transformer_lens_is_a_supported_version(self):
        # Mirrors circuitkit.backends._tl_compat's own version guard: prefer
        # __version__, fall back to packaging metadata (some installs, e.g.
        # certain Colab environments, ship without __version__ set).
        import transformer_lens

        version = getattr(transformer_lens, "__version__", "") or ""
        if not version:
            version = importlib.metadata.version("transformer_lens")
        from circuitkit.backends._tl_compat import _SUPPORTED_TL_VERSION_PREFIXES

        assert version.startswith(_SUPPORTED_TL_VERSION_PREFIXES), (
            f"the _tl_compat port targets transformer_lens {_SUPPORTED_TL_VERSION_PREFIXES}, "
            f"found {version!r}"
        )


class TestTransformersVersionAndArchitectures:
    def test_transformers_new_enough_for_all_three_architectures(self):
        import transformers

        # SmolLM3ForCausalLM needs >=4.53; Cohere2ForCausalLM needs >=4.48.
        # The higher bound covers both.
        assert _parse_version(transformers.__version__) >= _parse_version("4.53"), (
            "need transformers>=4.53 for SmolLM3 + Cohere2, found " f"{transformers.__version__}"
        )

    def test_cohere_cohere2_smollm3_architectures_import(self):
        from transformers.models.cohere import CohereForCausalLM  # noqa: F401
        from transformers.models.cohere2 import Cohere2ForCausalLM  # noqa: F401
        from transformers.models.smollm3 import SmolLM3ForCausalLM  # noqa: F401


class TestSupportLibraries:
    @pytest.mark.parametrize(
        "module_name",
        ["einops", "safetensors", "accelerate", "huggingface_hub", "numpy", "pytest"],
    )
    def test_support_library_importable(self, module_name):
        importlib.import_module(module_name)
