"""Process-isolated oracle for only previously selected Q-A-TS positions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from . import stage05


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class-name", required=True,
                        choices=stage05.DEVELOPMENT_CLASSES)
    parser.add_argument("--json-root", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--positions", required=True,
                        help="sorted comma-separated previously queried positions")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source = args.json_root / f"{args.class_name}.json"
    if not source.is_file() or stage05.sha256_file(source) != args.source_sha256:
        raise SystemExit("queried-label metadata SHA-256 mismatch")
    positions = [int(value) for value in args.positions.split(",") if value]
    if positions != sorted(set(positions)) or any(value < 0 for value in positions):
        raise SystemExit("query positions must be sorted, unique, and nonnegative")
    labels = stage05.test_labels_for_class(args.json_root, args.class_name)
    if any(value >= len(labels) for value in positions):
        raise SystemExit("query position is outside the official label array")
    digest = hashlib.sha256(json.dumps(positions, separators=(",", ":"))
                            .encode("ascii")).hexdigest()
    sys.stdout.write(json.dumps({"ready": True, "label_count": len(labels),
                                 "source_sha256": args.source_sha256,
                                 "allowed_positions_sha256": digest}) + "\n")
    sys.stdout.flush()
    cursor = 0
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("close") is True:
                return 0
            position = int(request["position"])
            if cursor >= len(positions) or position != positions[cursor]:
                raise ValueError("request is not the next frozen queried position")
            value = int(labels[position])
            if value not in (0, 1):
                raise ValueError("official queried label is not binary")
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
