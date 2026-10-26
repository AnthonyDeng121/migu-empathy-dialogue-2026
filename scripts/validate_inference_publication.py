#!/usr/bin/env python3
"""Validate a completed inference prediction/report publication; fail closed."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.inference_publication import validate_publication


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", required=True, type=Path)
    parser.add_argument("--performance-report", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = validate_publication(args.prediction, args.performance_report)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    print(f"PASS publication={report['publication']['id']} samples={report['samples']} sha256={report['output_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
