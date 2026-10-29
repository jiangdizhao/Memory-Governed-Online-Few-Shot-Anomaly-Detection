"""Process-isolated queried-label oracle for the Stage 0.75-A online replay.

This process reads the official labels but exposes only the next precommitted
queried position requested by the online runner.  It never writes labels to an
artifact or logs a response value.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from . import stage05


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--class-name", required=True,
                        choices=stage05.DEVELOPMENT_CLASSES)
    parser.add_argument("--json-root", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--query-positions", required=True,
                        help="Comma-separated label-blind Q-A-TS positions")
    parser.add_argument("--first-position", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source = args.json_root / f"{args.class_name}.json"
    if stage05.sha256_file(source) != args.source_sha256:
        raise SystemExit("label oracle source metadata SHA-256 mismatch")
    labels = stage05.test_labels_for_class(args.json_root, args.class_name)
    positions = [int(value) for value in args.query_positions.split(",") if value]
    if positions != sorted(set(positions)) or any(p < args.first_position for p in positions):
        raise SystemExit("invalid or nonmonotone frozen query positions")
    cursor = next((i for i, value in enumerate(positions)
                   if value >= args.first_position), len(positions))
    sys.stdout.write(json.dumps({"ready": True, "label_count": len(labels)}) + "\n")
    sys.stdout.flush()
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("close") is True:
                return 0
            position = int(request["position"])
            if cursor >= len(positions) or position != positions[cursor]:
                raise ValueError("request is not the next frozen queried position")
            if not 0 <= position < len(labels):
                raise ValueError("queried position falls outside official label array")
            value = int(labels[position])
            cursor += 1
            sys.stdout.write(json.dumps({"position": position, "label": value}) + "\n")
            sys.stdout.flush()
        except Exception as exc:
            sys.stdout.write(json.dumps({"error": f"{type(exc).__name__}: {exc}"}) + "\n")
            sys.stdout.flush()
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
