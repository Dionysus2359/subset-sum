"""Independent subset enumeration, feasibility, guarantees and input contracts."""

import itertools
import random
import unittest
from decimal import Decimal
from fractions import Fraction

from solvers import approximate_trim, exact_bitset, exact_dp, parse_epsilon, solve


def exhaustive_sums(values, budget):
    """Enumerate subsets of positions, independently of either DP recurrence."""
    if len(values) > 16:
        raise ValueError("the exhaustive oracle is limited to 16 items")
    reachable = set()
    for mask in range(1 << len(values)):
        subtotal = sum(value for index, value in enumerate(values) if mask & (1 << index))
        if subtotal <= budget:
            reachable.add(subtotal)
    return reachable


class ExactSolverTests(unittest.TestCase):
    methods = ("dp", "bitset")

    def test_fixed_examples(self):
        cases = [
            ([], 10, 0),
            ([3, 5], 0, 0),
            ([0, 0, 5], 4, 0),
            ([3], 6, 3),
            ([5, 5], 10, 10),
            ([6, 10, 15], 17, 16),
            ([2, 3, 5], 10, 10),
            ([1, 2], 100, 3),
            ([1000, 7, 4], 11, 11),
            ([60, 100, 150], 179, 160),
            ([104, 102, 201, 101], 308, 307),
            ([3, 4], 6, 4),  # Reaches the core: an ascending DP would return 6.
            ([5, 5, 7], 10, 10),  # Reaches the core and needs both equal items.
            ([10**100, 6, 10, 15], 17, 16),  # Filter before any bit shift.
            ([0, 0], 10, 0),
            ([8, 9], 7, 0),
        ]
        for values, budget, expected in cases:
            self.assertEqual(max(exhaustive_sums(values, budget)), expected)
            for method in self.methods:
                with self.subTest(values=values, budget=budget, method=method):
                    self.assertEqual(solve(values, budget, method), expected)

    def test_prepared_cores_do_not_reuse_an_item(self):
        for core in (exact_dp, exact_bitset):
            with self.subTest(core=core.__name__):
                self.assertEqual(core([3], 6), 3)
                self.assertEqual(core([3, 4], 6), 4)
                self.assertEqual(core([5, 5, 7], 10), 10)
                self.assertEqual(core([], 0), 0)

    def test_seeded_instances_against_exhaustive_search(self):
        rng = random.Random(319)
        for case in range(200):
            values = [rng.randint(0, 100) for _ in range(rng.randint(0, 12))]
            budget = rng.randint(0, 250)
            expected = max(exhaustive_sums(values, budget))
            shuffled = values.copy()
            rng.shuffle(shuffled)
            for ordering in (values, list(reversed(values)), shuffled):
                for method in self.methods:
                    with self.subTest(case=case, method=method, ordering=ordering):
                        self.assertEqual(solve(ordering, budget, method), expected)

    def test_complete_small_domain_against_exhaustive_search(self):
        # Repetition systematically covers duplicates, zeros, and item order.
        for length in range(5):
            for values in itertools.product(range(4), repeat=length):
                for budget in range(9):
                    expected = max(exhaustive_sums(values, budget))
                    for method in self.methods:
                        with self.subTest(values=values, budget=budget, method=method):
                            self.assertEqual(solve(values, budget, method), expected)

    def test_sixteen_item_oracle_case(self):
        values = list(range(1, 17))
        expected = max(exhaustive_sums(values, 67))
        for method in self.methods:
            self.assertEqual(solve(values, 67, method), expected)

    def test_input_is_unchanged(self):
        for method in self.methods:
            for values, budget in (([0, 15, 6, 10, 1000, 6], 17), ([0, 2, 1], 100)):
                original = values.copy()
                solve(values, budget, method)
                self.assertEqual(values, original)

    def test_ordered_sequence_inputs(self):
        for method in self.methods:
            self.assertEqual(solve((6, 10, 15), 17, method), 16)
            self.assertEqual(solve(range(1, 5), 6, method), 6)

    def test_invalid_budget_types(self):
        for budget in (True, False, 1.0, "10", None, Fraction(2), Decimal(2)):
            for method in self.methods:
                with self.subTest(budget=budget, method=method):
                    with self.assertRaisesRegex(TypeError, "budget"):
                        solve([], budget, method)

    def test_negative_budget(self):
        for method in self.methods:
            with self.assertRaisesRegex(ValueError, "budget"):
                solve([], -1, method)

    def test_invalid_item_types_before_trivial_returns(self):
        for bad in (True, False, 3.0, "3", None, Fraction(3), Decimal(3)):
            for budget in (0, 2, 100):
                for method in self.methods:
                    with self.subTest(bad=bad, budget=budget, method=method):
                        with self.assertRaisesRegex(TypeError, r"values\[1\]"):
                            solve([1, bad], budget, method)

    def test_negative_items_before_trivial_returns(self):
        for values, budget in (([-1], 0), ([1, -1], 100), ([1000, -1000], 1)):
            for method in self.methods:
                with self.assertRaisesRegex(ValueError, "nonnegative"):
                    solve(values, budget, method)

    def test_invalid_sequence_containers(self):
        for values in (None, 3, "123", b"123", bytearray([1]), {1, 2}, {"a": 1}, iter([1])):
            with self.subTest(values=values):
                with self.assertRaisesRegex(TypeError, "sequence"):
                    solve(values, 10)

    def test_method_validation_before_trivial_returns(self):
        for method in ("unknown", "DP", "", None, [], True):
            with self.subTest(method=method):
                with self.assertRaisesRegex(ValueError, "method"):
                    solve([], 0, method)

    def test_scaling_flag_requires_a_boolean(self):
        for flag in (1, "yes", None):
            with self.assertRaisesRegex(TypeError, "boolean"):
                solve([], 0, gcd_scale=flag)

    def test_exact_methods_do_not_need_epsilon(self):
        for method in self.methods:
            self.assertEqual(solve([6, 10, 15], 17, method, epsilon="unused"), 16)

    def test_oracle_preserves_duplicate_positions(self):
        self.assertEqual(exhaustive_sums([5, 5], 10), {0, 5, 10})
        with self.assertRaisesRegex(ValueError, "16"):
            exhaustive_sums([1] * 17, 10)


class ApproximationAndScalingTests(unittest.TestCase):
    epsilons = ("0.05", "0.10", "0.25", "0.50")

    def assert_guarantee(self, values, budget, epsilon, reachable):
        answer = solve(values, budget, "approx", epsilon)
        optimum = max(reachable)
        ratio = Fraction(epsilon)
        self.assertIn(answer, reachable)
        self.assertLessEqual(answer, optimum)
        self.assertGreaterEqual(ratio.denominator * answer,
                                (ratio.denominator - ratio.numerator) * optimum)

    def test_seeded_all_variants_and_permutations(self):
        rng = random.Random(320)
        for case in range(200):
            values = [rng.randint(0, 100) for _ in range(rng.randint(0, 12))]
            budget = rng.randint(0, 250)
            reachable = exhaustive_sums(values, budget)
            shuffled = values.copy()
            rng.shuffle(shuffled)
            for ordering in (values, list(reversed(values)), shuffled):
                for method in ("dp", "bitset"):
                    for scale in (False, True):
                        with self.subTest(case=case, method=method, scale=scale):
                            self.assertEqual(solve(ordering, budget, method, gcd_scale=scale), max(reachable))
                for epsilon in self.epsilons:
                    with self.subTest(case=case, epsilon=epsilon):
                        self.assert_guarantee(ordering, budget, epsilon, reachable)

    def test_fixed_feasibility_and_bounds(self):
        cases = [([], 10), ([3, 5], 0), ([0, 0, 5], 4), ([3], 6), ([5, 5], 10),
                 ([6, 10, 15], 17), ([2, 3, 5], 10), ([1, 2], 100),
                 ([1000, 7, 4], 11), ([60, 100, 150], 179),
                 ([104, 102, 201, 101], 308), ([5, 5, 7], 10)]
        for values, budget in cases:
            reachable = exhaustive_sums(values, budget)
            for epsilon in self.epsilons:
                self.assert_guarantee(values, budget, epsilon, reachable)
            for method in ("dp", "bitset"):
                self.assertEqual(solve(values, budget, method, gcd_scale=True), max(reachable))

    def test_literature_case_has_actual_error(self):
        values = [104, 102, 201, 101]
        self.assertEqual(solve(values, 308, "bitset"), 307)
        self.assertEqual(solve(values, 308, "approx", "0.40"), 302)

    def test_strict_trimming_at_exact_equality(self):
        # n=2, epsilon=.20: delta=.05, so 105 == 100*(1+delta).
        self.assertEqual(solve([100, 105], 106, "approx", "0.20"), 100)
        self.assertEqual(solve([100, 106], 107, "approx", "0.20"), 106)

    def test_core_decimal_and_fraction_agree(self):
        values = [104, 102, 201, 101]
        self.assertEqual(approximate_trim(values, 308, "0.40"), 302)
        self.assertEqual(approximate_trim(values, 308, Fraction(2, 5)), 302)
        self.assertEqual(approximate_trim([], 0, "0.10"), 0)

    def test_epsilon_invalid_before_trivial_returns(self):
        invalid = ("0", "0.0", "1", "1.0", "-0.1", "1.1", "", ".", "nan", "inf",
                   "1/10", "1e-1", " 0.1", "+0.1", "0.1 ", "0_1", "０.１", None,
                   0.1, True, Fraction(1, 10))
        for epsilon in invalid:
            for values, budget in (([], 0), ([1, 2], 100), ([5, 6], 7)):
                with self.subTest(epsilon=epsilon, values=values):
                    with self.assertRaisesRegex((TypeError, ValueError), "epsilon"):
                        solve(values, budget, "approx", epsilon)

    def test_decimal_epsilon_is_exact(self):
        self.assertEqual(parse_epsilon("0.10"), Fraction(1, 10))
        self.assertEqual(parse_epsilon(".25"), Fraction(1, 4))
        self.assertEqual(parse_epsilon("0.00000000000000000001"), Fraction(1, 10**20))
        self.assertEqual(parse_epsilon("0.99999999999999999999"), Fraction(10**20 - 1, 10**20))

    def test_gcd_uses_usable_items_and_floor_budget(self):
        for method in ("dp", "bitset"):
            self.assertEqual(solve([0, 1001, 60, 100, 150], 179, method, gcd_scale=True), 160)
            self.assertEqual(solve([6, 10, 15], 17, method, gcd_scale=True), 16)
            self.assertEqual(solve([5, 5, 7], 10, method, gcd_scale=True), 10)

    def test_scaling_family_invariant(self):
        values = [20, 21, 40, 63, 99]
        base = solve(values, 150, "bitset")
        for factor in (1, 10, 100):
            scaled = [factor * value for value in values]
            for method in ("dp", "bitset"):
                self.assertEqual(solve(scaled, factor * 150 + factor - 1, method, gcd_scale=True), factor * base)

    def test_approximation_rejects_gcd_even_on_trivial_inputs(self):
        with self.assertRaisesRegex(ValueError, "exact methods"):
            solve([], 0, "approx", gcd_scale=True)

    def test_input_validation_and_immutability_for_new_variants(self):
        configurations = [("approx", False), ("dp", True), ("bitset", True)]
        for method, scale in configurations:
            values = [0, 15, 6, 10, 1000, 6]
            original = values.copy()
            solve(values, 17, method, gcd_scale=scale)
            self.assertEqual(values, original)
            for bad_values, budget in (([True], 0), ([1, -2], 100), ([1.0], 0)):
                with self.assertRaises((TypeError, ValueError)):
                    solve(bad_values, budget, method, gcd_scale=scale)


if __name__ == "__main__":
    unittest.main()
