#!/usr/bin/env python3
"""Compare two deterministic slow-inference parity reports.

The reports must use the same schema. Timing and RSS fields are ignored only
when they are absent; callers should keep parity payloads free of timing data.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


def compare(left: Any, right: Any, path: str = "$", atol: float = 0.0, rtol: float = 0.0) -> list[str]:
    errors: list[str] = []

    if isinstance(left, bool) or isinstance(right, bool):
        if type(left) is not type(right) or left != right:
            errors.append(f"{path}: bool mismatch ({left!r} != {right!r})")
        return errors

    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if math.isnan(float(left)) != math.isnan(float(right)):
            errors.append(f"{path}: NaN mismatch")
            return errors
        if math.isinf(float(left)) != math.isinf(float(right)):
            errors.append(f"{path}: infinity mismatch ({left!r} != {right!r})")
            return errors
        if math.isnan(float(left)) or math.isinf(float(left)):
            return errors
        if not math.isclose(float(left), float(right), abs_tol=atol, rel_tol=rtol):
            errors.append(f"{path}: {left!r} != {right!r}")
        return errors

    if type(left) is not type(right):
        errors.append(f"{path}: type mismatch ({type(left).__name__} != {type(right).__name__})")
        return errors

    if isinstance(left, dict):
        left_keys, right_keys = set(left), set(right)
        for key in sorted(left_keys - right_keys):
            errors.append(f"{path}: missing key {key!r}")
        for key in sorted(right_keys - left_keys):
            errors.append(f"{path}: unexpected key {key!r}")
        for key in sorted(left_keys & right_keys):
            errors.extend(compare(left[key], right[key], f"{path}.{key}", atol, rtol))
        return errors

    if isinstance(left, list):
        if len(left) != len(right):
            errors.append(f"{path}: length mismatch ({len(left)} != {len(right)})")
            return errors
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            errors.extend(compare(left_item, right_item, f"{path}[{index}]", atol, rtol))
        return errors

    if left != right:
        errors.append(f"{path}: {left!r} != {right!r}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    args = parser.parse_args()

    if args.atol < 0 or args.rtol < 0:
        parser.error("--atol and --rtol must be non-negative")

    try:
        left = json.loads(args.left.read_text(encoding="utf-8"))
        right = json.loads(args.right.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    errors = compare(left, right, atol=args.atol, rtol=args.rtol)
    if errors:
        print(f"mismatch: {len(errors)} difference(s)", file=sys.stderr)
        for error in errors[:100]:
            print(f"  {error}", file=sys.stderr)
        if len(errors) > 100:
            print(f"  ... {len(errors) - 100} more", file=sys.stderr)
        return 1

    print("match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())