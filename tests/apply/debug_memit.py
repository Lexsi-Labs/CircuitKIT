"""Manual reproduction of test_memit_vs_rome_magnitude_parity for diagnostic logging.

Runs MEMIT on the same single fact the test uses, with INFO logging surfaced
to stdout. No test framework involved — just direct calls.
"""

import logging
import sys

# Configure logging FIRST, before any circuitkit import. The logger handlers
# are attached at module level in some files, and we want them to use our
# stdout handler from the start.
logging.basicConfig(
    level=logging.INFO,
    format="%(name)s | %(message)s",
    stream=sys.stdout,
    force=True,  # override any prior basicConfig
)
# Make sure circuitkit's loggers propagate to root (they do by default, but
# defensive in case any module sets propagate=False).
logging.getLogger("circuitkit").setLevel(logging.INFO)

from transformer_lens import (  # noqa: E402 - import after intentional pre-import setup
    HookedTransformer,
)

from circuitkit.applications.editing.memit_wrapper import (  # noqa: E402 - import after intentional pre-import setup
    MemitHandler,
)
from circuitkit.applications.editing.rome_wrapper import (  # noqa: E402 - import after intentional pre-import setup
    RomeHandler,
)


def main():
    print("=" * 70)
    print("Loading GPT-2 small...")
    print("=" * 70)
    model = HookedTransformer.from_pretrained("gpt2")
    model.cfg.use_hook_mlp_in = True
    model.eval()

    prompt = "The capital of France is"
    subject = "France"
    target = "Lyon"

    # Snapshot clean weights
    initial = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # ── ROME (control: known-good) ───────────────────────────────────────────
    print()
    print("=" * 70)
    print("ROME edit (control)")
    print("=" * 70)
    rome = RomeHandler(model)
    rome_result = rome.edit_single_fact(
        prompt=prompt,
        subject=subject,
        target=target,
        target_layer=5,
    )
    print(
        f"\nROME result: success={rome_result.success} "
        f"|edit|={rome_result.edit_magnitude:.4f} "
        f"conf_before={rome_result.confidence_before:.4e} "
        f"conf_after={rome_result.confidence_after:.4e}"
    )

    # Restore clean state
    model.load_state_dict(initial)

    # ── MEMIT (failing) ──────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("MEMIT edit (failing)")
    print("=" * 70)
    memit = MemitHandler(model)
    memit_results = memit.edit_multiple_facts(
        facts=[(prompt, subject, target)],
        target_layers=[5],
    )
    res = memit_results[0]
    print(
        f"\nMEMIT result: success={res.success} "
        f"|edit|={res.edit_magnitude:.4f} "
        f"conf_before={res.confidence_before:.4e} "
        f"conf_after={res.confidence_after:.4e}"
    )


if __name__ == "__main__":
    main()
