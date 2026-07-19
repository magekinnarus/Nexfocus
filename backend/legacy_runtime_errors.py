"""Neutral errors for retired runtime surfaces.

This module intentionally contains contracts only.  It must not grow a
compatibility API for any archived runtime family.
"""


class LegacyFluxArchivedError(NotImplementedError):
    """Raised when an archived legacy Flux Fill surface is invoked."""

    def __init__(
        self,
        message: str = (
            "Legacy Flux Fill has been archived. Use the active "
            "backend.flux_fill_v3 runtime."
        ),
    ) -> None:
        super().__init__(message)
