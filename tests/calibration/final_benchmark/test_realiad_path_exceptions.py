from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.calibration.final_benchmark import identity
from experiments.calibration.final_benchmark import realiad_path_exceptions as registered


def sha(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


class RegisteredRealIADPathExceptionTests(unittest.TestCase):
    def _case(self, kind: str):
        if kind == "fire_hood":
            category, image_key, position, code = (
                "fire_hood",
                "NG/HS/S0131/fire_hood__0131_NG_QS_C1_20260101000000.png",
                2790,
                "DIRECTORY_SESSION_CONTRADICTS_FILENAME_SESSION",
            )
        else:
            category, image_key, position, code = (
                "vcpill",
                "NG/BX/S141/vcpill__0141_NG_BX_C2_20260101000000.png",
                2856,
                "SAMPLE_DIRECTORY_GRAMMAR_REJECTS_OFFICIAL_PATH",
            )
        image_id = identity._image_identity_digest(image_key)
        group, camera = identity._realiad_identity_key(category, image_key)
        content = b"synthetic image bytes"
        annotation_sha = sha(f"official synthetic annotation:{category}")
        spec = {
            "image_identity_sha256": image_id,
            "category": category,
            "annotation_sha256": annotation_sha,
            "image_content_sha256": sha(content),
            "source_group_sha256": identity._source_group_digest(group),
            "camera_id": camera,
            "original_position": position,
            "validation_code": code,
        }
        return category, image_key, image_id, group, camera, position, content, annotation_sha, spec

    def _validate_case(self, kind: str):
        values = self._case(kind)
        category, image_key, image_id, group, camera, position, content, annotation_sha, spec = values
        manifest_sha = sha(f"synthetic registered manifest:{kind}")
        with tempfile.TemporaryDirectory() as directory:
            exact = Path(directory) / "images" / category / image_key
            exact.parent.mkdir(parents=True)
            exact.write_bytes(content)
            with patch.object(registered, "REGISTERED_EXCEPTIONS", {image_id: spec}), \
                    patch.object(registered, "REGISTERED_IDENTITY_MANIFEST_SHA256", manifest_sha):
                result = identity.validate_realiad_source_path(
                    path=exact, category=category, split="test",
                    annotation_sha256=annotation_sha, official_image_key=image_key,
                    official_group_key=group, image_identity_sha256=image_id,
                    source_group_sha256=spec["source_group_sha256"], camera_id=camera,
                    original_position=position, image_content_sha256=spec["image_content_sha256"],
                    relative_id=image_key, identity_manifest_sha256=manifest_sha)
            return values, result, exact

    def test_registered_fire_hood_session_directory_exception(self):
        values, result, _path = self._validate_case("fire_hood")
        self.assertEqual(result["path_exception"], values[-1]["validation_code"])
        self.assertEqual(result["relative_id"], values[1])

    def test_registered_vcpill_sample_directory_exception(self):
        values, result, _path = self._validate_case("vcpill")
        self.assertEqual(result["path_exception"], values[-1]["validation_code"])
        self.assertEqual(result["camera_id"], 2)

    def test_exception_rejects_wrong_bytes_position_annotation_or_path(self):
        values = self._case("fire_hood")
        category, image_key, image_id, group, camera, position, content, annotation_sha, spec = values
        manifest_sha = sha("synthetic registered manifest:fire_hood")
        with tempfile.TemporaryDirectory() as directory:
            exact = Path(directory) / "images" / category / image_key
            exact.parent.mkdir(parents=True)
            exact.write_bytes(content)
            with patch.object(registered, "REGISTERED_EXCEPTIONS", {image_id: spec}), \
                    patch.object(registered, "REGISTERED_IDENTITY_MANIFEST_SHA256", manifest_sha):
                bad_calls = [
                    {"annotation_sha256": sha("wrong annotation")},
                    {"original_position": position + 1},
                    {"image_content_sha256": sha("wrong registered bytes")},
                    {"identity_manifest_sha256": sha("wrong manifest")},
                ]
                for override in bad_calls:
                    args = dict(path=exact, category=category, split="test",
                        annotation_sha256=annotation_sha, official_image_key=image_key,
                        official_group_key=group, image_identity_sha256=image_id,
                        source_group_sha256=spec["source_group_sha256"], camera_id=camera,
                        original_position=position, image_content_sha256=spec["image_content_sha256"],
                        relative_id=image_key, identity_manifest_sha256=manifest_sha)
                    args.update(override)
                    with self.assertRaises(identity.IdentityAuditError):
                        identity.validate_realiad_source_path(**args)

                exact.write_bytes(b"tampered image bytes")
                with self.assertRaisesRegex(identity.IdentityAuditError, "content differs"):
                    identity.validate_realiad_source_path(path=exact, category=category, split="test",
                        annotation_sha256=annotation_sha, official_image_key=image_key,
                        official_group_key=group, image_identity_sha256=image_id,
                        source_group_sha256=spec["source_group_sha256"], camera_id=camera,
                        original_position=position, image_content_sha256=spec["image_content_sha256"],
                        relative_id=image_key, identity_manifest_sha256=manifest_sha)

    def test_no_unregistered_basename_fallback_and_registry_inventory(self):
        values = self._case("fire_hood")
        category, image_key, image_id, group, camera, position, content, annotation_sha, spec = values
        manifest_sha = sha("synthetic registered manifest:fire_hood")
        with tempfile.TemporaryDirectory() as directory:
            wrong_suffix = (Path(directory) / "images" / category / "NG" / "OTHER" / "S0131" /
                            image_key.rsplit("/", 1)[-1])
            wrong_suffix.parent.mkdir(parents=True)
            wrong_suffix.write_bytes(content)
            with patch.object(registered, "REGISTERED_EXCEPTIONS", {image_id: spec}), \
                    patch.object(registered, "REGISTERED_IDENTITY_MANIFEST_SHA256", manifest_sha):
                with self.assertRaisesRegex(identity.IdentityAuditError, "path suffix"):
                    identity.validate_realiad_source_path(path=wrong_suffix, category=category, split="test",
                        annotation_sha256=annotation_sha, official_image_key=image_key,
                        official_group_key=group, image_identity_sha256=image_id,
                        source_group_sha256=spec["source_group_sha256"], camera_id=camera,
                        original_position=position, image_content_sha256=spec["image_content_sha256"],
                        relative_id=image_key, identity_manifest_sha256=manifest_sha)
                with patch.object(registered, "REGISTERED_EXCEPTIONS", {}):
                    with self.assertRaises(identity.IdentityAuditError):
                        identity.validate_realiad_source_path(path=wrong_suffix, category=category, split="test",
                            annotation_sha256=annotation_sha, official_image_key=image_key,
                            official_group_key=group, image_identity_sha256=image_id,
                            source_group_sha256=spec["source_group_sha256"], camera_id=camera,
                            original_position=position, image_content_sha256=spec["image_content_sha256"],
                            relative_id=image_key, identity_manifest_sha256=manifest_sha)

        self.assertEqual(len(registered.REGISTERED_EXCEPTIONS), 15)
        self.assertEqual({row["category"] for row in registered.REGISTERED_EXCEPTIONS.values()},
                         {"fire_hood", "vcpill"})
        self.assertEqual(sum(row["category"] == "fire_hood"
                             for row in registered.REGISTERED_EXCEPTIONS.values()), 10)
        self.assertEqual(sum(row["category"] == "vcpill"
                             for row in registered.REGISTERED_EXCEPTIONS.values()), 5)


if __name__ == "__main__":
    unittest.main()
