"""Export and verify a compact, privacy-safe Stage 0 research audit package."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any


PACKAGE_FILES = (
    "gates.json", "test_results.json", "native_parity_smoke.json",
    "native_greedy_parity_matrix.json", "native_event_replay.json",
    "real_iad_metadata_preflight.json", "controller_method_cards.json",
)
DROP_KEYS = {"path", "absolute_path", "dataset_root", "json_root",
             "python_executable", "environment_prefix", "output_dir"}
ABSOLUTE_PATH = re.compile(
    r"(?i)(?<![a-z0-9])(?:[a-z]:[\\/]|\\\\[^\\]+\\[^\\]+|/(?:home|mnt|users)/)")


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sanitize(value: Any, key: str = "") -> Any:
    if key.casefold() in DROP_KEYS or "absolute_path" in key.casefold():
        return None
    if isinstance(value, dict):
        result = {}
        for child_key, child in value.items():
            normalized = child_key.casefold()
            if (normalized in DROP_KEYS or "absolute_path" in normalized or
                    "example_" in normalized or normalized.endswith("_sample_keys") or
                    normalized in {"relative_id", "sample_key", "official_sample_key"}):
                continue
            cleaned = _sanitize(child, child_key)
            if cleaned is not None:
                result[child_key] = cleaned
        return result
    if isinstance(value, (list, tuple)):
        return [_sanitize(item, key) for item in value]
    if isinstance(value, str):
        return "<REDACTED_LOCAL_PATH>" if ABSOLUTE_PATH.search(value) else value
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def _compact_tests(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: payload.get(key) for key in (
        "tests_run", "passed", "failed", "errors", "skipped",
        "duration_seconds", "passed_test_ids", "failures", "errors_detail",
        "skipped_detail")}


def export_audit(source_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    source, output = Path(source_dir), Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite audit package: {output}")
    required = set(PACKAGE_FILES) | {"checksums.json", "protocol_manifest.json",
                                    "source_hashes.json", "run_identity.json"}
    missing = sorted(name for name in required if not (source / name).is_file())
    if missing:
        raise FileNotFoundError(f"Runner artifact lacks required audit sources: {missing}")
    source_checksums = json.loads((source / "checksums.json").read_text(encoding="utf-8"))
    source_hashes = json.loads((source / "source_hashes.json").read_text(encoding="utf-8"))
    identity = json.loads((source / "run_identity.json").read_text(encoding="utf-8"))
    manifest = json.loads((source / "protocol_manifest.json").read_text(encoding="utf-8"))
    for name, expected in source_checksums.items():
        if not (source / name).is_file() or _sha_file(source / name) != expected:
            raise RuntimeError(f"Runner artifact checksum mismatch: {name}")

    output.mkdir(parents=True, exist_ok=False)
    for name in PACKAGE_FILES:
        payload = json.loads((source / name).read_text(encoding="utf-8"))
        if name == "test_results.json":
            payload = _compact_tests(payload)
        _write_json(output / name, _sanitize(payload))
    fingerprints = {
        "schema": "calibration.stage0.public_protocol_fingerprints.v1",
        "source_commit": source_hashes.get("source_commit"),
        "source_fingerprint_sha256": source_hashes.get("source_fingerprint_sha256"),
        "stage0_source_hashes": source_hashes.get("stage0_source_hashes", {}),
        "profile_sha256": manifest.get("profile_sha256"),
        "protocol_version": manifest.get("protocol_version"),
        "support_regime": manifest.get("support_regime"),
    }
    _write_json(output / "protocol_fingerprints.json", fingerprints)
    gates = json.loads((output / "gates.json").read_text(encoding="utf-8"))
    tests = json.loads((output / "test_results.json").read_text(encoding="utf-8"))
    replay = json.loads((output / "native_event_replay.json").read_text(encoding="utf-8"))
    matrix = json.loads((output / "native_greedy_parity_matrix.json").read_text(encoding="utf-8"))
    summary = [
        "# Stage 0 compact scientific audit", "",
        f"Runner status: **{manifest.get('overall_status', 'UNVERIFIED')}**.",
        "This package contains implementation evidence, not scientific validation.",
        "Stage 1 and Stage 2 were not run; no confirmation-class outcomes were opened.",
        "All Real-IAD scoring evidence is limited to the bounded development-class training-normal diagnostic.",
        "Synthetic binary labels in native replay are marked synthetic and are not Real-IAD outcomes.",
        "",
        "## Gate snapshot", "",
        *[f"- {name}: {row.get('status', 'UNVERIFIED')} — {row.get('reason', '')}"
          for name, row in gates.items()],
        "",
        f"Executed tests: {tests.get('tests_run', 0)}; passed {tests.get('passed', 0)}; "
        f"failed {tests.get('failed', 0)}; errors {tests.get('errors', 0)}; "
        f"skipped {tests.get('skipped', 0)}.",
        f"Synthetic native Greedy operator matrix: {matrix.get('status', 'UNVERIFIED')}.",
        f"Native event replay: {replay.get('status', 'UNVERIFIED')}; "
        f"all-controller matrix: {replay.get('all_controller_matrix_status', 'UNVERIFIED')}.",
        "",
        "## Provenance", "",
        "The compact evidence files are content-hashed in `checksums.json`. "
        "`audit_manifest.json` records the SHA-256 of the full local runner checksum manifest and the full source artifact files, "
        "so reviewers can verify the redacted export's relationship to that run.",
        "",
        "Every unavailable requirement remains BLOCKED, SKIPPED, or UNVERIFIED in the machine-readable gates.",
    ]
    (output / "scientific_audit_summary.md").write_text("\n".join(summary) + "\n",
                                                           encoding="utf-8")
    source_files = {name: _sha_file(source / name) for name in sorted(required)}
    _write_json(output / "audit_manifest.json", {
        "schema": "calibration.stage0.audit_export.v1",
        "run_key": identity.get("run_key"),
        "source_artifact_checksums_sha256": _sha_file(source / "checksums.json"),
        "source_artifact_file_sha256": source_files,
        "redaction_policy": "absolute local paths and example/sample identities removed; score vectors/features omitted",
        "overall_status": manifest.get("overall_status", "UNVERIFIED"),
    })
    checksums = {path.name: _sha_file(path)
                 for path in sorted(output.iterdir()) if path.is_file()}
    _write_json(output / "checksums.json", {
        "schema": "calibration.stage0.audit_checksums.v1",
        "files": checksums,
        "source_artifact_checksums_sha256": _sha_file(source / "checksums.json"),
    })
    verification = verify_audit(output, source_dir=source)
    return {"status": "PASS" if verification["valid"] else "FAIL",
            "output_dir": str(output), "files": sorted(checksums),
            "source_artifact_checksums_sha256": _sha_file(source / "checksums.json"),
            "verification": verification}


def verify_audit(package_dir: str | Path, *,
                 source_dir: str | Path | None = None) -> dict[str, Any]:
    package = Path(package_dir)
    check_path = package / "checksums.json"
    if not check_path.is_file():
        return {"valid": False, "reason": "checksums.json is absent"}
    checks = json.loads(check_path.read_text(encoding="utf-8"))
    files = checks.get("files", {})
    actual = {path.name for path in package.iterdir()
              if path.is_file() and path.name != "checksums.json"}
    if set(files) != actual:
        return {"valid": False, "reason": "package file set differs from checksum manifest"}
    for name, expected in files.items():
        if _sha_file(package / name) != expected:
            return {"valid": False, "reason": f"package checksum mismatch: {name}"}
    for path in package.iterdir():
        if path.is_file() and path.suffix.lower() in {".json", ".md", ".jsonl"}:
            content = path.read_text(encoding="utf-8")
            if ABSOLUTE_PATH.search(content):
                return {"valid": False, "reason": f"absolute local path found in {path.name}"}
    linked = False
    if source_dir is not None:
        source = Path(source_dir)
        manifest = json.loads((package / "audit_manifest.json").read_text(encoding="utf-8"))
        if _sha_file(source / "checksums.json") != manifest.get(
                "source_artifact_checksums_sha256"):
            return {"valid": False, "reason": "source runner checksum-manifest link mismatch"}
        for name, expected in manifest.get("source_artifact_file_sha256", {}).items():
            if not (source / name).is_file() or _sha_file(source / name) != expected:
                return {"valid": False, "reason": f"source artifact link mismatch: {name}"}
        linked = True
    return {"valid": True, "package_checksums_verified": True,
            "privacy_scan_passed": True, "source_artifact_link_verified": linked}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--export", action="store_true")
    mode.add_argument("--verify", action="store_true")
    parser.add_argument("--source-output-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.export:
        if args.source_output_dir is None:
            parser.error("--export requires --source-output-dir")
        result = export_audit(args.source_output_dir, args.output_dir)
    else:
        result = verify_audit(args.output_dir, source_dir=args.source_output_dir)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if result.get("status") == "PASS" or result.get("valid") else 1


if __name__ == "__main__":
    raise SystemExit(main())
