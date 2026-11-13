"""Solve one subset-sum instance from inline integers or a JSON file."""

import argparse
import json
from pathlib import Path
from collections.abc import Sequence

from solvers import solve


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--values", nargs="*", type=int, help="integer items; may be empty")
    source.add_argument("--input", type=Path, help='JSON file with "values" and "budget"')
    parser.add_argument("--budget", type=int, help="required with --values")
    parser.add_argument("--method", choices=("dp", "bitset", "approx"), default="dp")
    parser.add_argument("--epsilon", default="0.10", help="decimal error bound for approx (default 0.10)")
    parser.add_argument("--gcd-scale", action="store_true", help="scale usable items for exact methods")
    args = parser.parse_args(argv)

    if args.input is not None:
        if args.budget is not None:
            parser.error("--budget cannot be combined with --input; use the JSON budget")
        try:
            with args.input.open(encoding="utf-8") as stream:
                instance = json.load(stream)
        except (OSError, UnicodeError, ValueError) as error:
            parser.error(f"cannot read input: {error}")
        if not isinstance(instance, dict) or not {"values", "budget"} <= instance.keys():
            parser.error('input must be a JSON object containing "values" and "budget"')
        values, budget = instance["values"], instance["budget"]
    else:
        if args.budget is None:
            parser.error("--budget is required with --values")
        values, budget = args.values, args.budget

    try:
        answer = solve(values, budget, method=args.method, epsilon=args.epsilon, gcd_scale=args.gcd_scale)
    except (TypeError, ValueError) as error:
        parser.error(str(error))

    label = args.method + ("_gcd" if args.gcd_scale else "")
    print(f"method: {label}")
    print(f"budget: {budget}")
    print(f"best_sum: {answer}")
    if args.method == "approx":
        print(f"epsilon: {args.epsilon}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
