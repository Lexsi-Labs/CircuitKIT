import math


def count_at_fraction(n: int, fraction: float) -> int:
    """``floor(n * fraction)``, without losing a whole item to float error.

    A sparsity derived from a count rarely maps back exactly: with 612 nodes and
    ``sparsity = 1 - 20 / 612``, ``612 * (1 - sparsity)`` is 19.999..., which
    plain ``int()`` truncates to 19. The 1e-9 tolerance is far below one item.
    """
    return math.floor(n * fraction + 1e-9)
