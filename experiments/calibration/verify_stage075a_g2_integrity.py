"""Independent, read-only integrity audit for the historical Actual-B3 package.

This verifier deliberately does not import or execute the Actual-B3 runner. It
reads the published package, reconstructs the writer's transient five-entry
checksum manifest in memory, runs in-memory mutation-negative checks, and
writes a compact certificate to a new directory outside the package.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT = REPOSITORY_ROOT / "outputs/calibration/stage075a_b3_headroom_v1_20261030_verified2"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "docs/calibration/audit/stage075a_g2_integrity_cert_v2_20261030"
HISTORICAL_INCOMPLETE_BYTES = b"Actual-B3 replay/headroom diagnostic is incomplete.\n"
CHECKSUM_SCHEMA = "stage075a.b3_headroom.checksums.v1"
EXPECTED_CURRENT_ENTRIES = {
    "analysis.json",
    "b3_replay.json",
    "run_manifest.json",
    "source_provenance.json",
}
EXPECTED_PACKAGE_FILES = EXPECTED_CURRENT_ENTRIES | {"checksums.json", "complete.json"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_bytes(value: Any, *, newline: bool = False) -> bytes:
    """Match profiles.canonical_json plus the runner's _atomic_json newline."""
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return encoded + (b"\n" if newline else b"")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root).as_posix()
        result[relative] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    return result


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def flip_digest_hex(digest: str) -> str:
    if not digest:
        raise ValueError("Cannot mutate an empty digest")
    first = "0" if digest[0] != "0" else "1"
    return first + digest[1:]


def one_byte_flip(data: bytes) -> bytes:
    if not data:
        raise ValueError("Cannot mutate empty content")
    mutated = bytearray(data)
    mutated[len(mutated) // 2] ^= 1
    return bytes(mutated)


def require(condition: bool, label: str, checks: list[dict[str, Any]]) -> None:
    checks.append({"check": label, "status": "PASS" if condition else "FAIL"})


def run_negative_tests(
    *,
    artifact_bytes: dict[str, bytes],
    checksums: dict[str, str],
    run_manifest: dict[str, Any],
    complete: dict[str, Any],
    reconstructed_bytes: bytes,
) -> list[dict[str, str]]:
    """Prove the independent predicates reject one-byte in-memory mutations."""
    tests: list[dict[str, str]] = []

    replay = artifact_bytes["b3_replay.json"]
    replay_mutated = one_byte_flip(replay)
    tests.append({
        "test": "payload_byte_mutation_rejected_by_checksum",
        "status": "PASS" if sha256_bytes(replay_mutated) != checksums["b3_replay.json"] else "FAIL",
    })

    checksum_mutated = dict(checksums)
    checksum_mutated["analysis.json"] = flip_digest_hex(checksum_mutated["analysis.json"])
    all_match = all(
        sha256_bytes(artifact_bytes[name]) == expected
        for name, expected in checksum_mutated.items()
    )
    tests.append({
        "test": "checksum_entry_one_byte_mutation_rejected",
        "status": "PASS" if not all_match else "FAIL",
    })

    replay_link_mutated = flip_digest_hex(run_manifest["replay_sha256"])
    tests.append({
        "test": "run_manifest_replay_link_one_byte_mutation_rejected",
        "status": "PASS" if replay_link_mutated != sha256_bytes(replay) else "FAIL",
    })

    analysis_link_mutated = flip_digest_hex(run_manifest["analysis_sha256"])
    tests.append({
        "test": "run_manifest_analysis_link_one_byte_mutation_rejected",
        "status": "PASS" if analysis_link_mutated != sha256_bytes(artifact_bytes["analysis.json"]) else "FAIL",
    })

    provenance_link_mutated = flip_digest_hex(run_manifest["source_provenance_sha256"])
    tests.append({
        "test": "run_manifest_source_provenance_link_one_byte_mutation_rejected",
        "status": "PASS" if provenance_link_mutated != sha256_bytes(artifact_bytes["source_provenance.json"]) else "FAIL",
    })

    complete_link_mutated = flip_digest_hex(complete["checksums_manifest_sha256"])
    tests.append({
        "test": "complete_checksum_manifest_link_one_byte_mutation_rejected",
        "status": "PASS" if complete_link_mutated != sha256_bytes(reconstructed_bytes) else "FAIL",
    })

    reconstructed_mutated = one_byte_flip(reconstructed_bytes)
    tests.append({
        "test": "reconstructed_manifest_byte_mutation_rejected",
        "status": "PASS" if sha256_bytes(reconstructed_mutated) != complete["checksums_manifest_sha256"] else "FAIL",
    })
    return tests


def verify(artifact: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, str]], dict[str, dict[str, Any]]]:
    artifact = artifact.resolve(strict=True)
    before = inventory(artifact)
    names = set(before)
    package_file_set_exact = names == EXPECTED_PACKAGE_FILES

    artifact_bytes = {name: (artifact / name).read_bytes() for name in names}
    checksums_doc = json.loads(artifact_bytes["checksums.json"].decode("utf-8"))
    complete = json.loads(artifact_bytes["complete.json"].decode("utf-8"))
    run_manifest = json.loads(artifact_bytes["run_manifest.json"].decode("utf-8"))
    analysis = json.loads(artifact_bytes["analysis.json"].decode("utf-8"))
    replay = json.loads(artifact_bytes["b3_replay.json"].decode("utf-8"))
    provenance = json.loads(artifact_bytes["source_provenance.json"].decode("utf-8"))

    entries = checksums_doc.get("files", {})
    current_entries_exact = set(entries) == EXPECTED_CURRENT_ENTRIES
    current_entry_matches = {
        name: (name in artifact_bytes and sha256_bytes(artifact_bytes[name]) == digest)
        for name, digest in entries.items()
    }

    # The original writer's _checksums() returned sorted relative paths and
    # omitted checksums.json and complete.json. At its first pass, INCOMPLETE
    # was present; it is reconstructed here without touching the source tree.
    reconstructed_entries = dict(entries)
    reconstructed_entries["INCOMPLETE"] = sha256_bytes(HISTORICAL_INCOMPLETE_BYTES)
    reconstructed_manifest = {"schema": CHECKSUM_SCHEMA, "files": reconstructed_entries}
    reconstructed_bytes = canonical_json_bytes(reconstructed_manifest, newline=True)
    reconstructed_sha = sha256_bytes(reconstructed_bytes)

    raw_links = {
        "complete_to_run_manifest": complete.get("run_manifest_sha256") == sha256_bytes(artifact_bytes["run_manifest.json"]),
        "run_manifest_to_replay": run_manifest.get("replay_sha256") == sha256_bytes(artifact_bytes["b3_replay.json"]),
        "run_manifest_to_analysis": run_manifest.get("analysis_sha256") == sha256_bytes(artifact_bytes["analysis.json"]),
        "run_manifest_to_source_provenance": run_manifest.get("source_provenance_sha256") == sha256_bytes(artifact_bytes["source_provenance.json"]),
    }
    canonical_links = {
        "run_manifest_canonical_replay_digest": run_manifest.get("canonical_replay_digest") == sha256_bytes(canonical_json_bytes(replay)),
        "run_manifest_canonical_repeat_replay_digest": run_manifest.get("canonical_repeat_replay_digest") == sha256_bytes(canonical_json_bytes(replay)),
        "run_manifest_canonical_analysis_digest": run_manifest.get("canonical_analysis_digest") == sha256_bytes(canonical_json_bytes(analysis)),
        "run_manifest_canonical_repeat_analysis_digest": run_manifest.get("canonical_repeat_analysis_digest") == sha256_bytes(canonical_json_bytes(analysis)),
    }

    current_checksums_file_sha = sha256_bytes(artifact_bytes["checksums.json"])
    completion_linkage = {
        "current_checksums_sha256": current_checksums_file_sha,
        "complete_checksums_manifest_sha256": complete.get("checksums_manifest_sha256"),
        "current_manifest_matches_complete": current_checksums_file_sha == complete.get("checksums_manifest_sha256"),
        "reconstructed_five_entry_sha256": reconstructed_sha,
        "reconstructed_five_entry_matches_complete": reconstructed_sha == complete.get("checksums_manifest_sha256"),
        "complete_files_checked": complete.get("files_checked"),
        "current_entry_count": len(entries),
        "reconstructed_entry_count": len(reconstructed_entries),
    }

    checks: list[dict[str, Any]] = []
    require(package_file_set_exact, "target_has_expected_six_files", checks)
    require(checksums_doc.get("schema") == CHECKSUM_SCHEMA, "current_checksum_schema", checks)
    require(current_entries_exact, "current_manifest_has_expected_four_entries", checks)
    require(all(current_entry_matches.values()), "all_four_current_checksum_entries_match", checks)
    for name, passed in raw_links.items():
        require(passed, name, checks)
    for name, passed in canonical_links.items():
        require(passed, name, checks)
    require(complete.get("checksums_manifest_sha256") == reconstructed_sha,
            "reconstructed_historical_five_entry_manifest_matches_completion_pointer", checks)
    require(complete.get("files_checked") == 5 and len(entries) == 4,
            "completion_count_five_vs_current_count_four_reproduced", checks)

    negative_tests = run_negative_tests(
        artifact_bytes=artifact_bytes,
        checksums=entries,
        run_manifest=run_manifest,
        complete=complete,
        reconstructed_bytes=reconstructed_bytes,
    )
    for test in negative_tests:
        require(test["status"] == "PASS", test["test"], checks)

    after = inventory(artifact)
    require(before == after, "historical_package_hashes_unchanged_before_after", checks)
    status = "PASS" if all(check["status"] == "PASS" for check in checks) else "FAIL"

    result = {
        "schema": "stage075a.g2_independent_integrity_verification.v1",
        "status": status,
        "acceptance": "RESEARCHER_REVIEW_REQUIRED",
        "original_g2_gate_status": "BLOCKED",
        "verification_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_path": artifact.as_posix(),
        "historical_package_hashes_before": before,
        "historical_package_hashes_after": after,
        "historical_package_unchanged": before == after,
        "current_checksum_entry_matches": current_entry_matches,
        "raw_content_links": raw_links,
        "canonical_content_links": canonical_links,
        "completion_manifest_linkage": completion_linkage,
        "source_provenance_content": {
            "run_manifest_raw_sha256_link_verified": raw_links["run_manifest_to_source_provenance"],
            "embedded_run_manifest_sha256_is_upstream_reference": provenance.get("run_manifest_sha256"),
            "embedded_upstream_reference_is_not_current_run_manifest": provenance.get("run_manifest_sha256") != sha256_bytes(artifact_bytes["run_manifest.json"]),
        },
        "executed_source_provenance": {
            "exact_executed_runner_bytes_established": False,
            "reason": "The artifact finished before the runner source was committed, and the run manifest does not contain the executed runner source hash or a retained execution log.",
        },
        "checks": checks,
        "negative_test_count": len(negative_tests),
        "negative_tests_passed": all(test["status"] == "PASS" for test in negative_tests),
    }
    evidence = {
        "schema": "stage075a.g2_reconstructed_checksum_manifest_evidence.v1",
        "historical_writer_serialization": {
            "json": "json.dumps(sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)",
            "encoding": "UTF-8",
            "trailing_newline": True,
            "writer_path": "_atomic_json -> canonical_json(value) + '\\n'",
            "checksum_enumeration": "sorted recursive file paths; exclude checksums.json and complete.json",
        },
        "transient_entry": {
            "name": "INCOMPLETE",
            "content_utf8": HISTORICAL_INCOMPLETE_BYTES.decode("utf-8"),
            "content_hex": HISTORICAL_INCOMPLETE_BYTES.hex(),
            "byte_length": len(HISTORICAL_INCOMPLETE_BYTES),
            "sha256": sha256_bytes(HISTORICAL_INCOMPLETE_BYTES),
        },
        "reconstructed_manifest": reconstructed_manifest,
        "reconstructed_serialized_byte_length": len(reconstructed_bytes),
        "reconstructed_serialized_sha256": reconstructed_sha,
        "completion_pointer_sha256": complete.get("checksums_manifest_sha256"),
        "exact_match": reconstructed_sha == complete.get("checksums_manifest_sha256"),
        "current_manifest_sha256": current_checksums_file_sha,
        "current_manifest_entry_count": len(entries),
        "reconstructed_manifest_entry_count": len(reconstructed_entries),
    }
    return result, evidence, negative_tests, after


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def canonical_json_text(value: Any) -> str:
    return canonical_json_bytes(value, newline=True).decode("utf-8")


def source_analysis() -> str:
    return "\n".join([
        "# Historical writer source analysis",
        "",
        "Reviewed source: `experiments/calibration/stage075a_b3_headroom_runner.py` at reference commit `9c7ec2757e7eec731211bef2b3939a8e76cbe7cd`.",
        "",
        "- `_checksums` (lines 41-44) recursively hashes sorted files except `checksums.json` and `complete.json`; it does not exclude `INCOMPLETE`.",
        "- The runner creates `INCOMPLETE` with bytes `Actual-B3 replay/headroom diagnostic is incomplete.\\n` (line 53).",
        "- It writes the first `checksums.json` from that state (lines 128-130).",
        "- It removes its marker and then writes `complete.json`, recording the first checksum-manifest SHA-256 and `files_checked=len(checksums)` (lines 136-141).",
        "- It then runs `_checksums` again and rewrites `checksums.json` (lines 144-146). With the marker removed, the resulting manifest has four entries.",
        "- `_atomic_json` writes `canonical_json(value) + newline`; `canonical_json` sorts keys, uses compact separators, UTF-8 and disallows NaN.",
        "",
        "Independent reconstruction uses the four current verified entries plus the exact historical marker bytes in memory. The reconstructed five-entry canonical file has the exact SHA-256 referenced by `complete.json`. This certifies the checksum-linkage inconsistency and its reproducible writer sequence.",
        "",
        "## Provenance limitation",
        "",
        "The artifact records completion at `2026-10-29T19:12:36.101944+00:00` (2026-10-30 06:12:36 +11:00). Git history records runner source commit `1e8e66c512984ea34d74564b461a6a9072b7a02e` at `2026-10-30T06:21:44+11:00`, about nine minutes later; its parent is the artifact's `head_before_diagnostic` value `d699ad63a19e207e8b58e7cc42aee8c58abc48cc`.",
        "",
        "The run manifest does not carry a source hash for the executed runner, and no execution log or pre-refresh manifest is retained. Therefore the exact executed source bytes cannot be established from available evidence. The source analysis explains the observed output but does not claim proof of byte-identical execution source.",
        "",
    ])


def decision_candidate(result: dict[str, Any], evidence: dict[str, Any]) -> str:
    overall = result["status"]
    match = evidence["exact_match"]
    return f"""# G2 integrity decision candidate\n\n\
**Independent certificate: {overall}.** The current four payload checksum entries and run-manifest content links verify. The five-entry manifest reconstructed in memory, including the exact transient `INCOMPLETE` marker bytes, {'matches' if match else 'does not match'} the completion pointer. The one-byte mutation-negative checks and before/after historical hash comparison are recorded in `independent_tests.log` and `integrity_verification.json`.\n\n\
**Original G2 status: BLOCKED.** This certificate documents package integrity and reproduces the completion-manifest discrepancy. It does not itself close the original G2 gate.\n\n\
**Acceptance: RESEARCHER_REVIEW_REQUIRED.** Exact executed runner source bytes remain unverified because no executed-source hash or retained execution log is available and the repository source was committed after the recorded run. Human review is required for closure.\n"""


def build_audit_checksums(output: Path) -> dict[str, str]:
    return {
        path.relative_to(output).as_posix(): file_sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "checksums.json"
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    artifact = args.artifact.resolve(strict=True)
    output = args.output.resolve()
    if output == artifact or artifact in output.parents:
        parser.error("certificate output must not be inside the historical artifact")
    if output.exists():
        parser.error(f"certificate output already exists; refusing overwrite: {output}")

    result, evidence, negative_tests, after = verify(artifact)
    output.mkdir(parents=True, exist_ok=False)
    write_text(output / "source_code_analysis.md", source_analysis())
    write_text(output / "integrity_verification.json", canonical_json_text(result))
    write_text(output / "reconstructed_manifest_evidence.json", canonical_json_text(evidence))
    test_lines = [
        "Independent G2 integrity verification test log",
        f"Historical target hash inventory entries: {len(after)}",
        f"Historical target unchanged after verification: {result['historical_package_unchanged']}",
    ]
    test_lines.extend(f"{test['status']} {test['test']}" for test in negative_tests)
    test_lines.append(f"Overall certificate status: {result['status']}")
    test_lines.append(f"Acceptance status: {result['acceptance']}")
    write_text(output / "independent_tests.log", "\n".join(test_lines) + "\n")
    write_text(output / "integrity_decision_candidate.md", decision_candidate(result, evidence))
    checksums = build_audit_checksums(output)
    write_text(output / "checksums.json", canonical_json_text({"schema": "stage075a.g2_integrity_cert.checksums.v1", "files": checksums}))

    # Verify the compact certificate package itself after writing its manifest.
    audit_checksums_doc = read_json(output / "checksums.json")
    audit_integrity = all(
        file_sha256(output / name) == digest
        for name, digest in audit_checksums_doc["files"].items()
    )
    target_after_packaging = inventory(artifact)
    if target_after_packaging != result["historical_package_hashes_before"]:
        print("FAIL historical package changed during certificate writing", file=sys.stderr)
        return 2
    if not audit_integrity:
        print("FAIL certificate package checksum verification", file=sys.stderr)
        return 2
    print(f"certificate_status={result['status']}")
    print(f"acceptance={result['acceptance']}")
    print(f"historical_package_unchanged={result['historical_package_unchanged']}")
    print(f"negative_tests={len(negative_tests)} passed={all(t['status'] == 'PASS' for t in negative_tests)}")
    print(f"certificate_checksums_verified={audit_integrity}")
    print(f"output={output.as_posix()}")
    return 0 if result["status"] == "PASS" and audit_integrity else 1


if __name__ == "__main__":
    raise SystemExit(main())
