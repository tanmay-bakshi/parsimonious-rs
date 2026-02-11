"""General-purpose utilities.

This module mirrors the original ``parsimonious.utils`` API.

"""

from __future__ import annotations

import ast
from dataclasses import dataclass


class StrAndRepr:
    """Mix-in which gives the class the same ``__repr__`` and ``__str__``."""

    def __repr__(self) -> str:
        return str(self)


def evaluate_string(string: str) -> str | bytes:
    """Evaluate a Python string/bytes literal.

    :param string: A Python literal (e.g. ``"foo"`` or ``b"foo"``).
    :returns: The evaluated string/bytes.
    """

    return ast.literal_eval(string)


@dataclass(frozen=True, slots=True)
class Token(StrAndRepr):
    """A token for ``TokenGrammar`` inputs.

    Tokens must have a ``type`` attribute.

    :ivar type: The token type.
    """

    type: str

    def __str__(self) -> str:
        return '<Token "%s">' % (self.type,)

