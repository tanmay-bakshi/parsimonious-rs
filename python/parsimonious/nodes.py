"""Parse tree nodes and visitor utilities."""

from __future__ import annotations

from inspect import isfunction
from typing import Any, Callable, ClassVar

from parsimonious._parsimonious_rs import Node, RegexNode
from parsimonious.exceptions import UndefinedLabel, VisitationError


def rule(rule_string: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorate a `NodeVisitor` method to tie a grammar rule to it.

    :param rule_string: The right-hand side rule string.
    :returns: A decorator.
    """

    def decorator(method: Callable[..., Any]) -> Callable[..., Any]:
        setattr(method, "_rule", rule_string)
        return method

    return decorator


class RuleDecoratorMeta(type):
    """Metaclass which builds a default grammar from `@rule` methods."""

    def __new__(
        mcls, name: str, bases: tuple[type, ...], namespace: dict[str, Any]
    ) -> type:
        def unvisit(method_name: str) -> str:
            return method_name[6:] if method_name.startswith("visit_") else method_name

        methods = [
            v for _, v in namespace.items() if hasattr(v, "_rule") and isfunction(v)
        ]
        if len(methods) > 0:
            from parsimonious.grammar import Grammar

            methods.sort(key=lambda f: f.__code__.co_firstlineno)
            namespace["grammar"] = Grammar(
                "\n".join(f"{unvisit(m.__name__)} = {getattr(m, '_rule')}" for m in methods)
            )

        return super().__new__(mcls, name, bases, namespace)


class NodeVisitor(metaclass=RuleDecoratorMeta):
    """Depth-first visitor for parse trees."""

    grammar: ClassVar[Any | None] = None
    unwrapped_exceptions: ClassVar[tuple[type[BaseException], ...]] = ()

    def visit(self, node: Node) -> Any:
        """Visit a node.

        :param node: The node to visit.
        :returns: The visited value.
        :raises VisitationError: On errors thrown by visitor methods.
        """

        method = getattr(self, "visit_" + node.expr_name, self.generic_visit)
        try:
            return method(node, [self.visit(n) for n in node])
        except (VisitationError, UndefinedLabel):
            raise
        except Exception as exc:
            if isinstance(exc, self.unwrapped_exceptions):
                raise
            exc_class = type(exc)
            raise VisitationError(exc, exc_class, node) from exc

    def generic_visit(self, node: Node, visited_children: list[Any]) -> Any:
        """Default visitor method.

        :param node: The node being visited.
        :param visited_children: Results of visiting child nodes.
        :raises NotImplementedError: Always.
        """

        _ = visited_children
        raise NotImplementedError(
            "No visitor method was defined for this expression: %s" % node.expr.as_rule()
        )

    def parse(self, text: Any, pos: int = 0) -> Any:
        """Parse text with the visitor's default grammar and visit the result.

        :param text: The input.
        :param pos: Start position.
        :returns: The visited result.
        """

        return self._parse_or_match(text, pos, "parse")

    def match(self, text: Any, pos: int = 0) -> Any:
        """Match text with the visitor's default grammar and visit the result.

        :param text: The input.
        :param pos: Start position.
        :returns: The visited result.
        """

        return self._parse_or_match(text, pos, "match")

    def lift_child(self, node: Node, children: list[Any]) -> Any:
        """Lift the sole child of a node.

        :param node: The node.
        :param children: Visited children.
        :returns: The sole visited child.
        """

        _ = node
        (first_child,) = children
        return first_child

    def _parse_or_match(self, text: Any, pos: int, method_name: str) -> Any:
        if self.grammar is None:
            raise RuntimeError(
                "The {cls}.{method}() shortcut won't work because {cls} was never associated "
                "with a specific grammar. Fill out its `grammar` attribute, and try again.".format(
                    cls=self.__class__.__name__, method=method_name
                )
            )
        node = getattr(self.grammar, method_name)(text, pos=pos)
        return self.visit(node)

