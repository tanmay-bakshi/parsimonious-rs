"""Exception hierarchy for Parsimonious.

This mirrors the original Parsimonious exception types and messages.
"""

from textwrap import dedent
from typing import Any

from parsimonious.utils import StrAndRepr


class ParsimoniousError(Exception):
    """Base exception class for Parsimonious errors."""


class ParseError(StrAndRepr, ParsimoniousError):
    """Raised when parsing fails."""

    def __init__(self, text: Any, pos: int = -1, expr: Any | None = None):
        """Construct a ParseError.

        :param text: The input text (or token list for ``TokenGrammar``).
        :param pos: The position at which parsing failed.
        :param expr: The expression blamed for the failure.
        """

        self.text = text
        self.pos = pos
        self.expr = expr

    def __str__(self) -> str:
        expr = self.expr
        expr_name: str = ""
        if expr is not None:
            expr_name = getattr(expr, "name", "")

        if len(expr_name) > 0:
            rule_name = "'%s'" % expr_name
        else:
            rule_name = str(expr)

        window = self.text[self.pos : self.pos + 20]
        return "Rule %s didn't match at '%s' (line %s, column %s)." % (
            rule_name,
            window,
            self.line(),
            self.column(),
        )

    def line(self) -> int | None:
        """Return the 1-based line number where the expression ceased to match.

        :returns: The 1-based line number, or ``None`` for token inputs.
        """

        if isinstance(self.text, list):
            return None
        return self.text.count("\n", 0, self.pos) + 1

    def column(self) -> int:
        """Return the 1-based column where the expression ceased to match.

        :returns: The 1-based column number.
        """

        try:
            return self.pos - self.text.rindex("\n", 0, self.pos)
        except (ValueError, AttributeError):
            return self.pos + 1


class LeftRecursionError(ParseError):
    """Raised when the grammar contains left recursion."""

    def __str__(self) -> str:
        expr = self.expr
        rule_name = expr.name if (expr is not None and len(expr.name) > 0) else str(expr)
        window = self.text[self.pos : self.pos + 20]
        return dedent(
            f"""
            Left recursion in rule {rule_name!r} at {window!r} (line {self.line()}, column {self.column()}).

            Parsimonious is a packrat parser, so it can't handle left recursion.
            See https://en.wikipedia.org/wiki/Parsing_expression_grammar#Indirect_left_recursion
            for how to rewrite your grammar into a rule that does not use left-recursion.
            """
        ).strip()


class IncompleteParseError(ParseError):
    """Raised when parsing succeeds but does not consume the entire input."""

    def __str__(self) -> str:
        expr = self.expr
        expr_name: str = ""
        if expr is not None:
            expr_name = getattr(expr, "name", "")

        window = self.text[self.pos : self.pos + 20]
        return (
            "Rule '%s' matched in its entirety, but it didn't consume all the text. "
            "The non-matching portion of the text begins with '%s' (line %s, column %s)."
            % (expr_name, window, self.line(), self.column())
        )


class VisitationError(ParsimoniousError):
    """Raised when a ``NodeVisitor`` throws during visitation."""

    def __init__(self, exc: BaseException, exc_class: type[BaseException], node: Any):
        """Construct a VisitationError.

        :param exc: The underlying exception.
        :param exc_class: The class of the underlying exception.
        :param node: The node at which the error occurred.
        """

        self.original_class = exc_class
        super().__init__(
            "%s: %s\n\nParse tree:\n%s" % (exc_class.__name__, exc, node.prettily(error=node))
        )


class BadGrammar(StrAndRepr, ParsimoniousError):
    """Raised when a grammar definition is invalid."""


class UndefinedLabel(BadGrammar):
    """Raised when a grammar references an undefined rule."""

    def __init__(self, label: str):
        """Construct an UndefinedLabel.

        :param label: The missing rule name.
        """

        self.label = label

    def __str__(self) -> str:
        return 'The label "%s" was never defined.' % self.label

