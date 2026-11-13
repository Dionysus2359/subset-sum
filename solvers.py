"""Subset-sum optimization: return the largest reachable sum within a budget.

Use ``solve`` for validation and preparation. The exact core functions assume
prepared inputs: a sequence of positive integers, each at most the nonnegative
integer budget. Neither core mutates the supplied sequence.
"""

from collections.abc import Sequence
from fractions import Fraction
from math import gcd
import re


def parse_epsilon(epsilon: str) -> Fraction:
    """Accept decimal text without signs, whitespace, exponents or fractions."""
    if not isinstance(epsilon, str):
        raise TypeError("epsilon must be a decimal string, such as '0.10'")
    if not re.fullmatch(r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)", epsilon):
        raise ValueError("epsilon must be decimal text, such as '0.10'")
    ratio = Fraction(epsilon)
    if not 0 < ratio < 1:
        raise ValueError("epsilon must satisfy 0 < epsilon < 1")
    return ratio


def exact_dp(values: Sequence[int], budget: int) -> int:
    """Solve a prepared instance in O(nB) time and O(B) bytes of state.

    After each item, reachable[s] is one exactly when a subset of the processed
    positions sums to s. Complete every item pass, even if budget is reachable.
    """
    reachable = bytearray(budget + 1)
    reachable[0] = 1
    for value in values:
        # Descending sums read the previous-item state, using this item once.
        for subtotal in range(budget, value - 1, -1):
            reachable[subtotal] |= reachable[subtotal - value]
    return reachable.rfind(b"\x01")


def exact_bitset(values: Sequence[int], budget: int) -> int:
    """Solve a prepared instance using packed reachability in a Python integer.

    Bit s represents the same invariant as reachable[s] in exact_dp. Integer
    operations cost time proportional to their digit count, not constant time.
    """
    bits = 1
    mask = (1 << (budget + 1)) - 1
    for value in values:
        # Use the old bits on the right; mask sums that exceed the budget.
        bits = (bits | (bits << value)) & mask
    return bits.bit_length() - 1


def approximate_trim(
    values: Sequence[int], budget: int, epsilon: str | Fraction
) -> int:
    """Approximate a prepared instance using delta = epsilon / (2*n).

    Inputs obey the exact cores' preconditions. Decimal epsilon text is parsed
    exactly; solve may pass its already validated Fraction to avoid reparsing.
    Merging old sums and shifted sums is linear, without a per-pass sort.
    Every retained sum is feasible and the answer is at least (1-epsilon)*OPT.
    """
    ratio = epsilon if isinstance(epsilon, Fraction) else parse_epsilon(epsilon)
    if not 0 < ratio < 1:
        raise ValueError("epsilon must satisfy 0 < epsilon < 1")
    if not values:
        return 0
    p = ratio.numerator
    d = 2 * len(values) * ratio.denominator
    sums = [0]
    for value in values:
        shifted = [subtotal + value for subtotal in sums]
        left = right = 0
        retained = []
        while left < len(sums) or right < len(shifted):
            if right == len(shifted) or (left < len(sums) and sums[left] < shifted[right]):
                candidate = sums[left]
                left += 1
            elif left == len(sums) or shifted[right] < sums[left]:
                candidate = shifted[right]
                right += 1
            else:
                candidate = sums[left]
                left += 1
                right += 1  # Equal candidates represent one achievable sum.
            if candidate > budget:
                break
            # Strict comparison drops equality, using exact integer arithmetic.
            if not retained or d * candidate > (d + p) * retained[-1]:
                retained.append(candidate)
        sums = retained
    return sums[-1]


def solve(
    values: Sequence[int],
    budget: int,
    method: str = "dp",
    epsilon: str = "0.10",
    gcd_scale: bool = False,
) -> int:
    """Validate an instance and return its optimum without changing the input.

    Values must be an ordered sequence of nonnegative integers; the budget is
    a nonnegative integer. Booleans are rejected. Equal values are independent
    items. Zeros and oversized items are removed only after being validated.

    Methods are ``dp``, ``bitset``, and ``approx``. Approximation requires decimal
    epsilon text with 0 < epsilon < 1, including on trivial instances. GCD scaling
    is supported only for exact methods, which ignore epsilon.
    Core solver allocation, preparation, and result extraction all occur within
    this public call, which will be the benchmark measurement boundary.
    """
    if not isinstance(method, str) or method not in ("dp", "bitset", "approx"):
        raise ValueError("method must be 'dp', 'bitset', or 'approx'")
    if not isinstance(gcd_scale, bool):
        raise TypeError("gcd_scale must be a boolean")
    if gcd_scale and method == "approx":
        raise ValueError("GCD scaling is supported only for exact methods")
    ratio = parse_epsilon(epsilon) if method == "approx" else None
    if not isinstance(budget, int) or isinstance(budget, bool):
        raise TypeError("budget must be a nonnegative integer, not a boolean")
    if budget < 0:
        raise ValueError("budget must be nonnegative")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise TypeError("values must be an ordered sequence of nonnegative integers")

    prepared = []
    for index, value in enumerate(values):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"values[{index}] must be an integer, not a boolean")
        if value < 0:
            raise ValueError(f"values[{index}] must be nonnegative")
        if 0 < value <= budget:
            prepared.append(value)

    # Validate the entire original sequence before any trivial-case return.
    if budget == 0 or not prepared:
        return 0
    total = sum(prepared)
    if total <= budget:
        return total

    effective_budget = min(budget, total)
    if method == "approx":
        return approximate_trim(prepared, effective_budget, ratio)

    divisor = 1
    if gcd_scale:
        divisor = 0
        for value in prepared:
            divisor = gcd(divisor, value)
        if divisor > 1:
            prepared = [value // divisor for value in prepared]
            # A feasible sum is a multiple of divisor; round the budget down.
            effective_budget //= divisor
    solver = exact_dp if method == "dp" else exact_bitset
    return divisor * solver(prepared, effective_budget)
