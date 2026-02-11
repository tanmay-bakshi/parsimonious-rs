"""Grammar compilation and management."""

from __future__ import annotations

from typing import Any

from parsimonious._parsimonious_rs import Grammar as _Grammar
from parsimonious._parsimonious_rs import TokenGrammar as _TokenGrammar

# The grammar for parsing PEG grammar definitions (kept for compatibility).
rule_syntax: str = (
    r'''
    rules = _ rule*
    rule = label equals expression
    equals = "=" _
    literal = spaceless_literal _

    # So you can't spell a regex like `~"..." ilm`:
    spaceless_literal = ~"u?r?b?\"[^\"\\\\]*(?:\\\\.[^\"\\\\]*)*\""is /
                        ~"u?r?b?'[^'\\\\]*(?:\\\\.[^'\\\\]*)*'"is

    expression = ored / sequence / term
    or_term = "/" _ term+
    ored = term+ or_term+
    sequence = term term+
    not_term = "!" term _
    lookahead_term = "&" term _
    term = not_term / lookahead_term / quantified / atom
    quantified = atom quantifier
    atom = reference / literal / regex / parenthesized
    regex = "~" spaceless_literal ~"[ilmsuxa]*"i _
    parenthesized = "(" _ expression ")" _
    quantifier = ~r"[*+?]|\{\d*,\d+\}|\{\d+,\d*\}|\{\d+\}" _
    reference = label !equals

    # A subsequent equal sign is the only thing that distinguishes a label
    # (which begins a new rule) from a reference (which is just a pointer to a
    # rule defined somewhere else):
    label = ~"[a-zA-Z_][a-zA-Z_0-9]*(?![\"'])" _

    # _ = ~r"\s*(?:#[^\r\n]*)?\s*"
    _ = meaninglessness*
    meaninglessness = ~r"\s+" / comment
    comment = ~r"#[^\r\n]*"
    '''
)


class Grammar(_Grammar):
    """A collection of rules that describe a language."""


class TokenGrammar(_TokenGrammar):
    """A grammar over a token stream."""


rule_grammar: Grammar = Grammar(rule_syntax)


class LazyReference(str):
    """A lazy reference to a rule name (compatibility only)."""

    name: str = ""

    def resolve_refs(self, rule_map: dict[str, Any]) -> Any:
        """Resolve the reference in a rule map.

        :param rule_map: Mapping of rule names to expressions.
        :returns: The resolved expression.
        """

        seen: set[str] = set()
        cur: str = str(self)
        while True:
            if cur in seen:
                from parsimonious.exceptions import BadGrammar

                raise BadGrammar(f"Circular Reference resolving {self.name}={self}.")
            seen.add(cur)
            if cur not in rule_map:
                from parsimonious.exceptions import UndefinedLabel

                raise UndefinedLabel(cur)
            expr = rule_map[cur]
            if isinstance(expr, LazyReference):
                cur = str(expr)
                continue
            return expr

