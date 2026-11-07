"""Windows-only private storage for FS4 evidence and protected trust roots.

ACLs provide operational isolation from other unprivileged accounts; a local
administrator or the owning account can override them. Unsupported platforms,
unknown ACEs, reparse points, null DACLs and missing security APIs fail closed.
"""
from __future__ import annotations

import os
from pathlib import Path
import stat
from typing import Any


class FS4AccessControlError(RuntimeError):
    pass


def _windows() -> tuple[Any, Any, Any, Any]:
    if os.name != "nt":
        raise FS4AccessControlError("FS4 production storage requires Windows ACLs")
    try:
        import win32api
        import win32con
        import win32security
        import ntsecuritycon
        return win32api, win32con, win32security, ntsecuritycon
    except ImportError as exc:
        raise FS4AccessControlError("Windows security APIs unavailable") from exc


def current_user_sid() -> Any:
    api, con, security, _nt = _windows()
    token = security.OpenProcessToken(api.GetCurrentProcess(), con.TOKEN_QUERY)
    return security.GetTokenInformation(token, security.TokenUser)[0]


def _reject_reparse(path: Path) -> None:
    flags = getattr(path.lstat(), "st_file_attributes", 0)
    if path.is_symlink() or flags & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
        raise FS4AccessControlError("reparse points are forbidden for private FS4 storage")


def _sid_string(sid: Any) -> str:
    return _windows()[2].ConvertSidToStringSid(sid)


def set_private_acl(path: Path, *, directory: bool) -> None:
    """Restrict a new object to the current account and LocalSystem."""
    _api, con, security, nt = _windows()
    path = Path(path); _reject_reparse(path)
    acl = security.ACL()
    flags = (con.OBJECT_INHERIT_ACE | con.CONTAINER_INHERIT_ACE) if directory else 0
    for sid in (current_user_sid(), security.CreateWellKnownSid(security.WinLocalSystemSid, None)):
        acl.AddAccessAllowedAceEx(security.ACL_REVISION, flags, nt.FILE_ALL_ACCESS, sid)
    try:
        security.SetNamedSecurityInfo(str(path), security.SE_FILE_OBJECT,
            security.DACL_SECURITY_INFORMATION | security.PROTECTED_DACL_SECURITY_INFORMATION,
            None, None, acl, None)
    except Exception as exc:
        raise FS4AccessControlError("could not establish protected Windows DACL") from exc
    validate_private_acl(path, directory=directory)


def validate_private_acl(path: Path, *, directory: bool) -> None:
    """Validate owner, protected DACL, exact allowed principals and full rights."""
    _api, con, security, nt = _windows()
    path = Path(path); _reject_reparse(path)
    try:
        sd = security.GetNamedSecurityInfo(str(path), security.SE_FILE_OBJECT,
            security.OWNER_SECURITY_INFORMATION | security.DACL_SECURITY_INFORMATION)
        if _sid_string(sd.GetSecurityDescriptorOwner()) != _sid_string(current_user_sid()):
            raise FS4AccessControlError("private object owner differs from current account")
        if not (sd.GetSecurityDescriptorControl()[0] & security.SE_DACL_PROTECTED):
            raise FS4AccessControlError("private object DACL must not inherit permissions")
        acl = sd.GetSecurityDescriptorDacl()
        if acl is None:
            raise FS4AccessControlError("null DACL is not private")
        expected = {_sid_string(current_user_sid()),
                    _sid_string(security.CreateWellKnownSid(security.WinLocalSystemSid, None))}
        found = set()
        for i in range(acl.GetAceCount()):
            header, mask, sid = acl.GetAce(i)
            if header[0] != security.ACCESS_ALLOWED_ACE_TYPE or mask != nt.FILE_ALL_ACCESS:
                raise FS4AccessControlError("unexpected or non-full-access ACE")
            if directory and header[1] & (con.OBJECT_INHERIT_ACE | con.CONTAINER_INHERIT_ACE) != (con.OBJECT_INHERIT_ACE | con.CONTAINER_INHERIT_ACE):
                raise FS4AccessControlError("private directory does not protect children")
            found.add(_sid_string(sid))
        if found != expected or acl.GetAceCount() != len(expected):
            raise FS4AccessControlError("private DACL grants unexpected principals")
    except FS4AccessControlError:
        raise
    except Exception as exc:
        raise FS4AccessControlError("could not validate Windows security descriptor") from exc


class WindowsPrivateStore:
    """Non-overwriting directory with validation before and after each access."""
    def __init__(self, root: Path, *, create: bool) -> None:
        self.root = Path(root).absolute()
        for component in reversed((self.root, *self.root.parents)):
            if component.exists():
                _reject_reparse(component)
        if create:
            self.root.mkdir(parents=False, exist_ok=False)
            set_private_acl(self.root, directory=True)
        validate_private_acl(self.root, directory=True)

    def create_directory(self, name: str) -> "WindowsPrivateStore":
        return WindowsPrivateStore(self._child(name, must_exist=False), create=True)

    def _child(self, name: str, *, must_exist: bool) -> Path:
        if (not isinstance(name, str) or not name or Path(name).name != name or name in (".", "..") or
                any(char in name for char in ("/", "\\", ":")) or name.endswith((" ", "."))):
            raise FS4AccessControlError("private object name must be a single relative component")
        validate_private_acl(self.root, directory=True)
        path = self.root / name
        if must_exist:
            _reject_reparse(path)
        return path

    def write(self, name: str, content: bytes) -> Path:
        path = self._child(name, must_exist=False)
        with path.open("xb") as out:
            set_private_acl(path, directory=False)
            out.write(content); out.flush(); os.fsync(out.fileno())
        validate_private_acl(path, directory=False)
        return path

    def read(self, name: str) -> bytes:
        path = self._child(name, must_exist=True)
        validate_private_acl(path, directory=False)
        data = path.read_bytes()
        validate_private_acl(path, directory=False)
        return data
