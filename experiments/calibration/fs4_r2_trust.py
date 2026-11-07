"""Signed FS4 grants with a separately provisioned Windows trust anchor.

Production does not accept a caller-provided verification function or key.
An administrator must independently provision the fixed read-only authority
file before any future scientific run. No private signing key is stored here.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from .fs4_r2_security import FS4AccessControlError, _reject_reparse, _sid_string, _windows

AUTHORITY_FILE = Path(r"C:\ProgramData\FS4\trust\authorities.json")
ROLES = frozenset(("experiment_manifest", "stage_b_grant", "stage_b_result"))


class FS4TrustError(RuntimeError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _validate_anchor_object(path: Path) -> None:
    """Only System/Administrators may own or modify the provisioned anchor."""
    _api, con, security, nt = _windows()
    _reject_reparse(path)
    sd = security.GetNamedSecurityInfo(str(path), security.SE_FILE_OBJECT,
        security.OWNER_SECURITY_INFORMATION | security.DACL_SECURITY_INFORMATION)
    system = _sid_string(security.CreateWellKnownSid(security.WinLocalSystemSid, None))
    admins = _sid_string(security.CreateWellKnownSid(security.WinBuiltinAdministratorsSid, None))
    if _sid_string(sd.GetSecurityDescriptorOwner()) not in {system, admins}:
        raise FS4AccessControlError("trust anchor must be owned by System or Administrators")
    if not (sd.GetSecurityDescriptorControl()[0] & security.SE_DACL_PROTECTED):
        raise FS4AccessControlError("trust anchor must have a protected DACL")
    acl = sd.GetSecurityDescriptorDacl()
    if acl is None:
        raise FS4AccessControlError("trust anchor has a null DACL")
    write_bits = (nt.FILE_WRITE_DATA | nt.FILE_APPEND_DATA | nt.FILE_DELETE_CHILD | nt.FILE_WRITE_EA |
                  nt.FILE_WRITE_ATTRIBUTES | con.DELETE | con.WRITE_DAC | con.WRITE_OWNER |
                  con.GENERIC_WRITE | con.GENERIC_ALL)
    for i in range(acl.GetAceCount()):
        header, mask, sid = acl.GetAce(i)
        principal = _sid_string(sid)
        if header[0] != security.ACCESS_ALLOWED_ACE_TYPE:
            raise FS4AccessControlError("trust anchor contains an unexpected ACE")
        if principal not in {system, admins} and mask & write_bits:
            raise FS4AccessControlError("run account can modify the trust anchor")


def load_production_authorities() -> dict[str, bytes]:
    """Read only the fixed, independently administered production trust root."""
    path = AUTHORITY_FILE
    if not path.is_file():
        raise FS4TrustError("independently provisioned production authority file is absent")
    for item in (path.parent.parent, path.parent, path):
        _validate_anchor_object(item)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("schema") != "FS4_AUTHORITIES_V1" or set(value.get("keys", {})) != ROLES:
            raise ValueError("invalid authority schema")
        keys = {role: base64.b64decode(value["keys"][role], validate=True) for role in ROLES}
        if any(len(key) != 32 for key in keys.values()) or len(set(keys.values())) != len(ROLES):
            raise ValueError("authority keys must be distinct Ed25519 public keys")
        for item in (path.parent.parent, path.parent, path):
            _validate_anchor_object(item)
        return keys
    except (ValueError, KeyError, TypeError) as exc:
        raise FS4TrustError("invalid independently provisioned authority file") from exc


def verify_signed(document: Mapping[str, Any], *, role: str, authorities: Mapping[str, bytes]) -> dict[str, Any]:
    """Verify a domain-separated Ed25519 document using a role-specific key."""
    if role not in ROLES or set(document) != {"payload", "signature"} or not isinstance(document["payload"], dict):
        raise FS4TrustError("invalid signed document")
    try:
        signature = base64.b64decode(document["signature"], validate=True)
        key = Ed25519PublicKey.from_public_bytes(authorities[role])
        key.verify(signature, f"FS4-R2:{role}\0".encode() + canonical(document["payload"]))
    except (InvalidSignature, ValueError, KeyError, TypeError) as exc:
        raise FS4TrustError("signed document failed independent authority verification") from exc
    return dict(document["payload"])


def verify_production_signed(document: Mapping[str, Any], *, role: str) -> dict[str, Any]:
    return verify_signed(document, role=role, authorities=load_production_authorities())
