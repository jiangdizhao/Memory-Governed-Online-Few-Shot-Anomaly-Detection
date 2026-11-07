"""FS4 availability and fatal integrity boundaries.

Only explicit availability exceptions are recoverable. Fatal exceptions invalidate
the current condition and must never be turned into an ordinary threshold hold.
"""


class FS4InsufficientEvidence(Exception):
    """A legitimate evidence gate is not yet satisfied."""


class FS4RecoverableAvailabilityError(Exception):
    """A temporary external resource is unavailable; retry is permitted."""


class FS4FatalIntegrityError(RuntimeError):
    """The condition cannot safely continue or produce scientific metrics."""


class FS4ProvenanceIntegrityError(FS4FatalIntegrityError, ValueError):
    """Source, receipt, prototype or feature identity is inconsistent."""


class FS4NativeScoringIntegrityError(FS4FatalIntegrityError):
    """Native scoring failed or returned an invalid result."""


class FS4StateMutationError(FS4FatalIntegrityError):
    """A supposedly read-only scoring operation altered a live state."""
