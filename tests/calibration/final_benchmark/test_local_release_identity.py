import hashlib
import json
import unittest

from experiments.calibration.final_benchmark.entrypoint import (
    EntryPointError,
    validate_release_identity,
)


SNAPSHOT = "d44a059bf998231fcdc2a296b97ca45544a2554075e1f92b4dfc15c89a5fe4e2"
DECISION = "9d1a5e4f7d0f27e6e40743586675d868707a0afda236ec288c60e96a2527c6cb"
GATE0 = "a34b870fd4700bfe970e459d2963a602f04b1341"


def canonical_sha(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class LocalReleaseIdentityTests(unittest.TestCase):
    def _binding(self):
        identifier = f"realiad-local-snapshot-v1:{SNAPSHOT}"
        release_sha = hashlib.sha256(
            ("Gate0-local-release-v1\0realiad\0" + identifier).encode("utf-8")
        ).hexdigest()
        identity_manifest = {
            "schema": "final-benchmark.identity-manifest.v1",
            "dataset_id": "realiad",
            "category_count": 30,
            "categories": [{"category": f"cat_{i}"} for i in range(30)],
            "identity_records": [],
            "local_snapshot_sha256": SNAPSHOT,
            "official_release_sha256": release_sha,
        }
        role_manifest = {
            "dataset_id": "realiad",
            "category": "cat_0",
            "support_seed": 0,
            "dataset_release_sha256": release_sha,
            "selected": [],
        }
        release_identity = {
            "schema": "final-benchmark.dataset-release-identity.v1",
            "status": "GATE0_CERTIFIED",
            "gate0_freeze_commit": GATE0,
            "dataset_id": "realiad",
            "dataset_release_sha256": release_sha,
            "identity_manifest_sha256": canonical_sha(identity_manifest),
            "role_manifest_sha256": canonical_sha(role_manifest),
            "provenance_kind": "RESEARCHER_APPROVED_LOCAL_SNAPSHOT",
            "local_snapshot_sha256": SNAPSHOT,
            "local_snapshot_identifier": identifier,
            "researcher_decision_sha256": DECISION,
            "publisher_authenticated": False,
        }
        return identity_manifest, role_manifest, release_identity, release_sha

    def test_approved_domain_separated_local_snapshot_digest_is_accepted(self):
        identity_manifest, role_manifest, release_identity, expected = self._binding()
        self.assertEqual(
            validate_release_identity(identity_manifest=identity_manifest,
                                      role_manifest=role_manifest,
                                      release_identity=release_identity),
            expected,
        )
        self.assertEqual(len(expected), 64)
        self.assertFalse(release_identity["publisher_authenticated"])

    def test_mismatched_or_unapproved_local_identity_is_rejected(self):
        identity_manifest, role_manifest, release_identity, _ = self._binding()
        candidate = dict(release_identity, status="CANDIDATE_NOT_APPROVED")
        with self.assertRaises(EntryPointError):
            validate_release_identity(identity_manifest=identity_manifest,
                                      role_manifest=role_manifest,
                                      release_identity=candidate)
        bad_release = dict(release_identity, dataset_release_sha256="0" * 64)
        with self.assertRaises(EntryPointError):
            validate_release_identity(identity_manifest=identity_manifest,
                                      role_manifest=role_manifest,
                                      release_identity=bad_release)
        bad_snapshot = dict(release_identity, local_snapshot_sha256="1" * 64)
        with self.assertRaises(EntryPointError):
            validate_release_identity(identity_manifest=identity_manifest,
                                      role_manifest=role_manifest,
                                      release_identity=bad_snapshot)


if __name__ == "__main__":
    unittest.main()
