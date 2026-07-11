from circuitkit.data.corruption import InstructionSwap, audit_instruction_swap_degeneracy
from circuitkit.data.normalized import ContrastiveRecord


class _ToyTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        if text == " ":
            return [0]
        if text.startswith(" "):
            return [0, self._token_id(text.strip())]
        return [self._token_id(text)]

    def _token_id(self, text):
        return {"yes": 11, "no": 12, "maybe": 13}.get(text, 50 + len(text))


def test_instruction_swap_degeneracy_audit_reports_same_first_token_fraction():
    strategy = InstructionSwap()
    clean = ContrastiveRecord(
        record_id="0",
        clean_prompt="Write a short poem",
        clean_answer=" yes",
    )
    paired = strategy.apply(clean)
    non_degenerate = ContrastiveRecord(
        record_id="1",
        clean_prompt="Explain the rule",
        clean_answer=" yes",
        corrupt_prompt="Explain the rule",
        corrupt_answer=" no",
    )

    audit = audit_instruction_swap_degeneracy([paired, non_degenerate], _ToyTokenizer())

    assert audit["total"] == 2
    assert audit["paired"] == 2
    assert audit["same_first_token"] == 1
    assert audit["same_first_token_frac"] == 0.5
    assert audit["kept"] == 1
    assert audit["dropped"] == 1
    assert audit["skipped_unpaired"] == 0
