"""End-to-end chat-template checks on real instruction-tuned models.

Parametrized over two model families (gemma and Llama-3) so the chat-template
corrections are verified family-agnostically. Each pins the contract that the
``_chat`` unit tests assert with fakes:

  * an ``"auto"`` downstream task (boolq) resolves to *templated* — its
    dataloader is flagged templated and the prompts carry the model's chat
    template prefix;
  * MMLU's dataloader is likewise flagged templated (regression guard for the
    builder that silently dropped the flag);
  * the EAP backend's real pair-tokenization cross-aligns the clean/corrupt
    pair to a shared length with exactly one BOS per row (no double-BOS — the
    chat template already renders its own BOS);
  * an ``"off"`` task (winogrande, a cloze task) stays raw on the same model;
  * a full EAP circuit discovery runs end to end and yields real scores.

Slow (loads multi-GB models); skipped automatically when transformer-lens is
missing, CUDA is unavailable, or a gated model cannot be downloaded.
"""

import re

import pytest

torch = pytest.importorskip("torch")
transformer_lens = pytest.importorskip("transformer_lens")

import circuitkit.selection as S  # noqa: E402  import after importorskip guard
from circuitkit.backends.eap.eap_utils import tokenize_batch_pair  # noqa: E402
from circuitkit.tasks.bootstrap import _bootstrap_builtin_tasks  # noqa: E402
from circuitkit.tasks.registry import get_task  # noqa: E402

pytestmark = [pytest.mark.slow, pytest.mark.integration]

_MODELS = ["google/gemma-2-2b-it", "meta-llama/Llama-3.2-1B-Instruct"]
_PROBE = "CK_PROBE"


def _cfg(model):
    """Discovery config keyed to the loaded model."""
    return {
        "model_name": model.cfg.model_name,
        "num_examples": 16,
        "max_batches": 2,
        "batch_size": 4,
        "algorithm": "eap",
        "level": "node",
    }


def _strip_volatile_date(s: str) -> str:
    """Redact the current date some chat templates embed (e.g. Llama-3's
    ``Today Date: 03 Jul 2026``). Without this, prefix comparisons are flaky
    across midnight and across cached dataset builds: the templated prompt in
    the dataloader may carry the build-day's date while ``_template_prefix``
    re-renders today's."""
    return re.sub(r"Today Date: \d{1,2} \w{3} \d{4}", "Today Date: <DATE>", s)


def _template_prefix(model):
    """The chat template's user-turn prefix (family-agnostic chat marker)."""
    probe = model.tokenizer.apply_chat_template(
        [{"role": "user", "content": _PROBE}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return _strip_volatile_date(probe.split(_PROBE)[0])


@pytest.fixture(scope="module", params=_MODELS)
def instruct_model(request):
    """Load an instruct model configured for EAP attribution, or skip."""
    model_name = request.param
    if not torch.cuda.is_available():
        pytest.skip("instruct-model integration test needs CUDA")
    _bootstrap_builtin_tasks()
    try:
        model = transformer_lens.HookedTransformer.from_pretrained(
            model_name, device="cuda", dtype="bfloat16"
        )
    except (OSError, ConnectionError, ValueError) as e:  # offline / gated
        pytest.skip(f"{model_name} unavailable: {e}")
    # Match the grid scripts' EAP config: split inputs, ungroup GQA.
    model.cfg.use_attn_result = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_hook_mlp_in = True
    model.cfg.ungroup_grouped_query_attention = True
    assert model.tokenizer.chat_template is not None, f"{model_name} must be a chat model"
    return model


def test_auto_task_resolves_templated(instruct_model):
    """boolq ("auto") templates on an instruct model and carries the chat prefix."""
    boolq = get_task("boolq")
    assert boolq.chat_template_mode == "auto"
    dl = boolq.build_dataloader(instruct_model, _cfg(instruct_model), "cuda")
    assert getattr(dl, "templated", False) is True
    clean = list(next(iter(dl))[0])
    assert _strip_volatile_date(clean[0]).startswith(_template_prefix(instruct_model))


def test_pair_tokenization_single_bos(instruct_model):
    """The EAP pair-tokenizer cross-aligns clean/corrupt with exactly one BOS."""
    boolq = get_task("boolq")
    dl = boolq.build_dataloader(instruct_model, _cfg(instruct_model), "cuda")
    side = getattr(dl, "pair_padding_side", None)
    templated = getattr(dl, "templated", False)
    batch = next(iter(dl))
    clean, corrupted = list(batch[0]), list(batch[1])

    ct, kt, _, _, _, _ = tokenize_batch_pair(
        instruct_model, clean, corrupted, pair_padding_side=side, templated=templated
    )
    assert ct.shape == kt.shape, "clean/corrupt must be cross-aligned to a shared length"

    bos_id = instruct_model.tokenizer.bos_token_id
    for row in list(ct) + list(kt):
        assert int((row == bos_id).sum()) == 1, "exactly one BOS per row — no double-BOS"


def test_mmlu_dataloader_marked_templated(instruct_model):
    """MMLU ("auto") must flag its dataloader ``templated`` — regression guard
    for the bug where the mmlu/glue/generic builders dropped the flag and the
    EAP backend then double-prepended BOS onto already-templated prompts."""
    mmlu = get_task("mmlu")
    assert mmlu.chat_template_mode == "auto"
    dl = mmlu.build_dataloader(instruct_model, _cfg(instruct_model), "cuda")
    assert getattr(dl, "templated", False) is True


def test_off_task_stays_raw(instruct_model):
    """winogrande ("off", a cloze task) is never templated, even on a chat model."""
    wg = get_task("winogrande")
    assert wg.chat_template_mode == "off"
    dl = wg.build_dataloader(instruct_model, _cfg(instruct_model), "cuda")
    assert getattr(dl, "templated", False) is False
    assert not _strip_volatile_date(list(next(iter(dl))[0])[0]).startswith(
        _template_prefix(instruct_model)
    )


def test_eap_discovery_runs_on_instruct(instruct_model):
    """A full EAP discovery on the templated instruct task yields real scores."""
    try:
        import circuitkit.selection.eap_selector  # noqa: F401  registers "eap"
    except ImportError as e:
        pytest.skip(f"experiments eap selector unavailable: {e}")

    scores = S.get_selector("eap")(instruct_model, "boolq", _cfg(instruct_model))
    vals = [float(v) for v in scores.values()]
    assert vals, "discovery returned no scores"
    assert all(v == v and abs(v) != float("inf") for v in vals), "all scores must be finite"
    # Non-degenerate: heads/MLPs are genuinely ranked, not all tied.
    assert len({round(v, 8) for v in vals}) > 10
