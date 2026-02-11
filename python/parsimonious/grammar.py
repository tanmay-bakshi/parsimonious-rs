"""A convenience which constructs expression trees from an easy-to-read syntax.

This module largely mirrors the original Parsimonious implementation, while
delegating parsing and matching to the Rust-backed expression types.
"""

from collections import OrderedDict
from textwrap import dedent
from typing import Any

from parsimonious.exceptions import BadGrammar, UndefinedLabel
from parsimonious.expressions import (
    Literal,
    Lookahead,
    Not,
    OneOf,
    OneOrMore,
    Optional,
    Quantifier,
    Regex,
    Sequence,
    TokenMatcher,
    ZeroOrMore,
    expression,
    is_callable,
)
from parsimonious.nodes import NodeVisitor
from parsimonious.utils import evaluate_string


class Grammar(OrderedDict):
    """A collection of rules that describe a language."""

    default_rule: Any | None

    def __init__(self, rules: str = "", **more_rules: Any):
        """Construct a grammar.

        :param rules: A string of production rules, one per line.
        :param more_rules: Additional keyword rules. Values may be Expressions
            or custom rule callables.
        """

        decorated_custom_rules: dict[str, Any] = {
            k: (expression(v, k, self) if is_callable(v) else v)
            for k, v in more_rules.items()
        }

        exprs, first = self._expressions_from_rules(rules, decorated_custom_rules)
        super().__init__(exprs.items())
        self.default_rule = first

    def default(self, rule_name: str) -> "Grammar":
        """Return a new Grammar whose default rule is ``rule_name``.

        :param rule_name: The rule name to make the default.
        :returns: A shallow copy with the default rule updated.
        """

        new = self._copy()
        new.default_rule = new[rule_name]
        return new

    def _copy(self) -> "Grammar":
        """Return a shallow copy of this grammar.

        :returns: A shallow copy.
        """

        new = Grammar.__new__(Grammar)
        super(Grammar, new).__init__(self.items())
        new.default_rule = self.default_rule
        return new

    def _expressions_from_rules(
        self, rules: str, custom_rules: dict[str, Any]
    ) -> tuple[OrderedDict[str, Any], Any | None]:
        """Compile rules into expressions.

        :param rules: The rule string.
        :param custom_rules: A mapping of custom rules which override string rules.
        :returns: A pair of (rule_map, default_rule).
        """

        tree = rule_grammar.parse(rules)
        return RuleVisitor(custom_rules).visit(tree)

    def parse(self, text: Any, pos: int = 0) -> Any:
        """Parse text with the default rule.

        :param text: The input text (or token list for ``TokenGrammar``).
        :param pos: The index at which to start parsing.
        :returns: A parse tree node.
        """

        self._check_default_rule()
        assert self.default_rule is not None
        return self.default_rule.parse(text, pos=pos)

    def match(self, text: Any, pos: int = 0) -> Any:
        """Match text with the default rule without consuming to EOF.

        :param text: The input text (or token list for ``TokenGrammar``).
        :param pos: The index at which to start matching.
        :returns: A parse tree node.
        """

        self._check_default_rule()
        assert self.default_rule is not None
        return self.default_rule.match(text, pos=pos)

    def _check_default_rule(self) -> None:
        """Raise RuntimeError if there is no default rule defined."""

        if self.default_rule is None:
            raise RuntimeError(
                "Can't call parse() on a Grammar that has no default rule. Choose a specific rule instead, "
                "like some_grammar['some_rule'].parse(...)."
            )

    def __str__(self) -> str:
        """Return a rule string that would reconstitute this grammar."""

        exprs: list[Any] = []
        if self.default_rule is not None:
            exprs.append(self.default_rule)
        exprs.extend(expr for expr in self.values() if expr is not self.default_rule)
        return "\n".join(expr.as_rule() for expr in exprs)

    def __repr__(self) -> str:
        return "Grammar({!r})".format(str(self))


class TokenGrammar(Grammar):
    """A Grammar which takes a list of pre-lexed tokens instead of text."""

    def _expressions_from_rules(
        self, rules: str, custom_rules: dict[str, Any]
    ) -> tuple[OrderedDict[str, Any], Any | None]:
        tree = rule_grammar.parse(rules)
        return TokenRuleVisitor(custom_rules).visit(tree)


class BootstrappingGrammar(Grammar):
    """Grammar used to recognize the textual rules that describe other grammars."""

    def _expressions_from_rules(
        self, rule_syntax_text: str, custom_rules: dict[str, Any]
    ) -> tuple[OrderedDict[str, Any], Any | None]:
        comment = Regex(r"#[^\r\n]*", name="comment")
        meaninglessness = OneOf(Regex(r"\s+"), comment, name="meaninglessness")
        underscore = ZeroOrMore(meaninglessness, name="_")
        equals = Sequence(Literal("="), underscore, name="equals")
        label = Sequence(Regex(r"[a-zA-Z_][a-zA-Z_0-9]*"), underscore, name="label")
        reference = Sequence(label, Not(equals), name="reference")
        quantifier = Sequence(Regex(r"[*+?]"), underscore, name="quantifier")
        spaceless_literal = Regex(
            r'u?r?b?"[^"\\]*(?:\\.[^"\\]*)*"',
            ignore_case=True,
            dot_all=True,
            name="spaceless_literal",
        )
        literal = Sequence(spaceless_literal, underscore, name="literal")
        regex = Sequence(
            Literal("~"),
            literal,
            Regex("[ilmsuxa]*", ignore_case=True),
            underscore,
            name="regex",
        )
        atom = OneOf(reference, literal, regex, name="atom")
        quantified = Sequence(atom, quantifier, name="quantified")

        term = OneOf(quantified, atom, name="term")
        not_term = Sequence(Literal("!"), term, underscore, name="not_term")
        term.members = (not_term,) + term.members

        sequence = Sequence(term, OneOrMore(term), name="sequence")
        or_term = Sequence(Literal("/"), underscore, OneOrMore(term), name="or_term")
        ored = Sequence(OneOrMore(term), OneOrMore(or_term), name="ored")
        expression_expr = OneOf(ored, sequence, term, name="expression")
        rule = Sequence(label, equals, expression_expr, name="rule")
        rules = Sequence(underscore, OneOrMore(rule), name="rules")

        rule_tree = rules.parse(rule_syntax_text)
        return RuleVisitor().visit(rule_tree)


# The grammar for parsing PEG grammar definitions.
rule_syntax: str = (
    r"""
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
    """
)


class LazyReference(str):
    """A lazy reference to a rule, resolved after compiling all rules."""

    name: str = ""

    def resolve_refs(self, rule_map: dict[str, Any]) -> Any:
        """Resolve this reference within a rule map.

        :param rule_map: Mapping of rule names to expressions.
        :returns: The resolved expression.
        :raises BadGrammar: If a circular reference is detected.
        :raises UndefinedLabel: If a label is not defined.
        """

        seen: set["LazyReference"] = set()
        cur: Any = self
        while True:
            if cur in seen:
                raise BadGrammar(f"Circular Reference resolving {self.name}={self}.")
            seen.add(cur)
            try:
                cur = rule_map[str(cur)]
            except KeyError as exc:
                raise UndefinedLabel(str(cur)) from exc
            if isinstance(cur, LazyReference):
                continue
            return cur

    def _as_rhs(self) -> str:
        return "<LazyReference to %s>" % self


class RuleVisitor(NodeVisitor):
    """Turn a parsed grammar definition into a map of expressions."""

    quantifier_classes: dict[str, Any] = {"?": Optional, "*": ZeroOrMore, "+": OneOrMore}

    visit_expression = visit_term = visit_atom = NodeVisitor.lift_child

    def __init__(self, custom_rules: dict[str, Any] | None = None):
        """Construct a visitor.

        :param custom_rules: Custom rules to override string rules.
        """

        self.custom_rules: dict[str, Any] = custom_rules if custom_rules is not None else {}
        self._last_literal_node_and_type: tuple[Any, type[Any]] | None = None

    def visit_parenthesized(self, node: Any, parenthesized: list[Any]) -> Any:
        left_paren, _, expression_expr, right_paren, _ = parenthesized
        _ = left_paren
        _ = right_paren
        return expression_expr

    def visit_quantifier(self, node: Any, quantifier: list[Any]) -> Any:
        symbol, _ = quantifier
        _ = node
        return symbol

    def visit_quantified(self, node: Any, quantified: list[Any]) -> Any:
        atom, quantifier = quantified
        _ = node
        try:
            return self.quantifier_classes[quantifier.text](atom)
        except KeyError:
            parts = quantifier.text[1:-1].split(",")
            if len(parts) == 1:
                min_match = max_match = int(parts[0])
            else:
                min_match = int(parts[0]) if len(parts[0]) > 0 else 0
                max_match = int(parts[1]) if len(parts[1]) > 0 else float("inf")
            return Quantifier(atom, min=min_match, max=max_match)

    def visit_lookahead_term(self, node: Any, lookahead_term: list[Any]) -> Any:
        ampersand, term, _ = lookahead_term
        _ = ampersand
        _ = node
        return Lookahead(term)

    def visit_not_term(self, node: Any, not_term: list[Any]) -> Any:
        exclamation, term, _ = not_term
        _ = exclamation
        _ = node
        return Not(term)

    def visit_rule(self, node: Any, rule: list[Any]) -> Any:
        label, equals, expression_expr = rule
        _ = equals
        _ = node
        expression_expr.name = label
        return expression_expr

    def visit_sequence(self, node: Any, sequence: list[Any]) -> Any:
        term, other_terms = sequence
        _ = node
        return Sequence(term, *other_terms)

    def visit_ored(self, node: Any, ored: list[Any]) -> Any:
        first_term, other_terms = ored
        _ = node
        if len(first_term) == 1:
            first_term = first_term[0]
        else:
            first_term = Sequence(*first_term)
        return OneOf(first_term, *other_terms)

    def visit_or_term(self, node: Any, or_term: list[Any]) -> Any:
        slash, _, terms = or_term
        _ = slash
        _ = node
        if len(terms) == 1:
            return terms[0]
        return Sequence(*terms)

    def visit_label(self, node: Any, label: list[Any]) -> str:
        name, _ = label
        _ = node
        return name.text

    def visit_reference(self, node: Any, reference: list[Any]) -> LazyReference:
        label, not_equals = reference
        _ = not_equals
        _ = node
        return LazyReference(label)

    def visit_regex(self, node: Any, regex: list[Any]) -> Any:
        tilde, literal, flags, _ = regex
        _ = tilde
        _ = node
        flags_text = flags.text.upper()
        pattern = literal.literal
        return Regex(
            pattern,
            ignore_case="I" in flags_text,
            locale="L" in flags_text,
            multiline="M" in flags_text,
            dot_all="S" in flags_text,
            unicode="U" in flags_text,
            verbose="X" in flags_text,
            ascii="A" in flags_text,
        )

    def visit_spaceless_literal(self, spaceless_literal: Any, visited_children: list[Any]) -> Any:
        _ = visited_children
        literal_value = evaluate_string(spaceless_literal.text)
        if self._last_literal_node_and_type is not None:
            last_node, last_type = self._last_literal_node_and_type
            if last_type != type(literal_value):
                raise BadGrammar(
                    dedent(
                        f"""\
                        Found {last_node.text} ({last_type}) and {spaceless_literal.text} ({type(literal_value)}) string literals.
                        All strings in a single grammar must be of the same type.
                        """
                    )
                )

        self._last_literal_node_and_type = spaceless_literal, type(literal_value)
        return Literal(literal_value)

    def visit_literal(self, node: Any, literal: list[Any]) -> Any:
        spaceless_literal, _ = literal
        _ = node
        return spaceless_literal

    def generic_visit(self, node: Any, visited_children: list[Any]) -> Any:
        if len(visited_children) > 0:
            return visited_children
        return node

    def visit_rules(self, node: Any, rules_list: list[Any]) -> tuple[OrderedDict[str, Any], Any | None]:
        _, rules = rules_list
        _ = node

        rule_map: OrderedDict[str, Any] = OrderedDict((expr.name, expr) for expr in rules)
        rule_map.update(self.custom_rules)

        for name, rule in list(rule_map.items()):
            if hasattr(rule, "resolve_refs"):
                rule_map[name] = rule.resolve_refs(rule_map)

        if isinstance(rules, list) and len(rules) > 0:
            return rule_map, rule_map[rules[0].name]
        return rule_map, None


class TokenRuleVisitor(RuleVisitor):
    """A visitor which builds expression trees meant for token streams."""

    def visit_spaceless_literal(self, spaceless_literal: Any, visited_children: list[Any]) -> Any:
        _ = visited_children
        return TokenMatcher(evaluate_string(spaceless_literal.text))

    def visit_regex(self, node: Any, regex: list[Any]) -> Any:
        _ = node
        _ = regex
        raise BadGrammar(
            "Regexes do not make sense in TokenGrammars, since TokenGrammars operate on pre-lexed tokens rather than characters."
        )


# Bootstrap to level 1...
rule_grammar = BootstrappingGrammar(rule_syntax)
# ...and then to level 2.
rule_grammar = Grammar(rule_syntax)

