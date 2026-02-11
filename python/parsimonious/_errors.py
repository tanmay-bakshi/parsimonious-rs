"""Internal helpers for error construction.

These utilities are intentionally not part of the public API.

"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class _DummyNode:
    """A minimal node-like object used to construct `VisitationError`."""

    def prettily(self, *, error: object | None = None) -> str:
        """Return a placeholder tree rendering.

        :param error: Unused; present for compatibility with ``Node.prettily``.
        :returns: A placeholder string.
        """

        _ = error
        return "<no parse tree available>"


DUMMY_NODE: _DummyNode = _DummyNode()

