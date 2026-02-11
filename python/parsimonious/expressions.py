"""Expression types.

The core expression engine is implemented in Rust and re-exported here.

"""

from __future__ import annotations

from inspect import getfullargspec, isfunction, ismethod, ismethoddescriptor
from typing import Any

from parsimonious._parsimonious_rs import (
    Expression,
    Literal,
    Lookahead,
    OneOf,
    Quantifier,
    Regex,
    Sequence,
    TokenMatcher,
)


def is_callable(value: Any) -> bool:
    """Return whether a value should be treated as a custom rule callable.

    :param value: The value to test.
    :returns: True if callable.
    """

    criteria = [isfunction, ismethod, ismethoddescriptor]
    return any(criterion(value) for criterion in criteria)


def expression(callable: Any, rule_name: str, grammar: Any) -> Expression:
    """Turn a plain callable into an `Expression`.

    The callable must take either 2 args (text, pos) or 5 args
    (text, pos, cache, error, grammar).

    :param callable: A Python callable implementing a custom rule.
    :param rule_name: The name of the rule.
    :param grammar: The grammar the rule is part of.
    :returns: A Rust-backed expression.
    :raises RuntimeError: If the callable signature is invalid.
    """

    if ismethoddescriptor(callable) and hasattr(callable, "__func__"):
        callable = callable.__func__

    num_args = len(getfullargspec(callable).args)
    if ismethod(callable):
        num_args -= 1

    if num_args == 2:
        arity = 2
    elif num_args == 5:
        arity = 5
    else:
        raise RuntimeError(
            "Custom rule functions must take either 2 or 5 arguments, not %s."
            % num_args
        )

    return Expression.custom(rule_name, callable, arity, grammar)

