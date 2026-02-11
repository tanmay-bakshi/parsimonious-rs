//! Core Rust expression engine for parsimonious-rs.
//!
//! This crate provides a PEG-style expression API inspired by Python
//! Parsimonious. Grammars are built from expression values and then parsed.

#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use regex::Regex;
use thiserror::Error;

/// Parse tree node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Node {
    /// Expression that produced this node.
    pub expression: Expression,
    /// Start byte offset.
    pub start: usize,
    /// End byte offset.
    pub end: usize,
    /// Child nodes.
    pub children: Vec<Node>,
}

impl Node {
    /// Return the matched slice.
    ///
    /// :param input: Input that was parsed.
    /// :returns: Matched text slice.
    pub fn text<'a>(&self, input: &'a str) -> &'a str {
        input.get(self.start..self.end).unwrap_or("")
    }
}

/// Parsing errors.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum ParseError {
    /// The expression did not match.
    #[error("rule {rule:?} did not match at byte {position}")]
    NoMatch { rule: String, position: usize },
    /// The expression matched but did not consume the full input.
    #[error("rule {rule:?} matched but did not consume all input at byte {position}")]
    Incomplete { rule: String, position: usize },
    /// Left recursion was detected.
    #[error("left recursion in rule {rule:?} at byte {position}")]
    LeftRecursion { rule: String, position: usize },
    /// The requested starting position is invalid for the given input length.
    #[error("invalid start position {position} for input length {len}")]
    InvalidPosition { position: usize, len: usize },
}

/// PEG expression.
#[derive(Clone)]
pub struct Expression {
    inner: Arc<ExpressionInner>,
}

impl PartialEq for Expression {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl Eq for Expression {}

impl Hash for Expression {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.inner).hash(state);
    }
}

impl Debug for Expression {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.as_rule())
    }
}

#[derive(Clone)]
struct ExpressionInner {
    name: String,
    kind: ExpressionKind,
}

#[derive(Clone)]
enum ExpressionKind {
    Literal(String),
    Regex(Regex),
    Sequence(Vec<Expression>),
    OneOf(Vec<Expression>),
    Lookahead {
        member: Expression,
        negative: bool,
    },
    Quantifier {
        member: Expression,
        min: usize,
        max: Option<usize>,
    },
}

#[derive(Clone, Copy)]
enum CacheSlot {
    Unknown,
    InProgress,
    Done(Option<usize>),
}

struct NativeNode {
    expression: Expression,
    start: usize,
    end: usize,
    children: Vec<usize>,
}

#[derive(Default)]
struct ErrorTracker {
    position: usize,
    expression: Option<Expression>,
}

impl ErrorTracker {
    fn update(&mut self, expression: &Expression, position: usize) {
        if position < self.position {
            return;
        }
        let has_named_expr = self
            .expression
            .as_ref()
            .map(|expr| expr.name().is_empty() == false)
            .unwrap_or(false);
        if expression.name().is_empty() && has_named_expr {
            return;
        }

        self.position = position;
        self.expression = Some(expression.clone());
    }

    fn to_error(&self, fallback: &Expression) -> ParseError {
        let expr = self.expression.as_ref().unwrap_or(fallback);
        ParseError::NoMatch {
            rule: expr.rule_name(),
            position: self.position,
        }
    }
}

struct Parser<'a> {
    input: &'a str,
    len: usize,
    cache: HashMap<usize, Vec<CacheSlot>>,
    arena: Vec<NativeNode>,
    error: ErrorTracker,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            input,
            len: input.len(),
            cache: HashMap::new(),
            arena: Vec::new(),
            error: ErrorTracker::default(),
        }
    }

    fn slots_for_expr_mut(&mut self, expr: &Expression) -> &mut Vec<CacheSlot> {
        let expr_id = expr.id();
        self.cache.entry(expr_id).or_insert_with(|| {
            let mut slots = Vec::with_capacity(self.len + 1);
            slots.resize_with(self.len + 1, || CacheSlot::Unknown);
            slots
        })
    }

    fn push_node(
        &mut self,
        expression: &Expression,
        start: usize,
        end: usize,
        children: Vec<usize>,
    ) -> usize {
        let idx = self.arena.len();
        self.arena.push(NativeNode {
            expression: expression.clone(),
            start,
            end,
            children,
        });
        idx
    }

    fn node_end(&self, idx: usize) -> usize {
        self.arena[idx].end
    }

    fn match_core(
        &mut self,
        expression: &Expression,
        pos: usize,
    ) -> Result<Option<usize>, ParseError> {
        if pos > self.len {
            return Ok(None);
        }

        match self.slots_for_expr_mut(expression)[pos] {
            CacheSlot::Unknown => {
                self.slots_for_expr_mut(expression)[pos] = CacheSlot::InProgress;
            }
            CacheSlot::InProgress => {
                return Err(ParseError::LeftRecursion {
                    rule: expression.rule_name(),
                    position: pos,
                });
            }
            CacheSlot::Done(result) => {
                return Ok(result);
            }
        }

        let result = self.uncached_match(expression, pos)?;
        self.slots_for_expr_mut(expression)[pos] = CacheSlot::Done(result);

        if result.is_none() {
            self.error.update(expression, pos);
        }
        Ok(result)
    }

    fn uncached_match(
        &mut self,
        expression: &Expression,
        pos: usize,
    ) -> Result<Option<usize>, ParseError> {
        match expression.kind() {
            ExpressionKind::Literal(literal) => {
                let Some(rest) = self.input.get(pos..) else {
                    return Ok(None);
                };
                if rest.starts_with(literal) {
                    let end = pos + literal.len();
                    let idx = self.push_node(expression, pos, end, Vec::new());
                    return Ok(Some(idx));
                }
                Ok(None)
            }
            ExpressionKind::Regex(regex) => {
                let found = regex.find_at(self.input, pos);
                if let Some(found) = found {
                    if found.start() == pos {
                        let idx = self.push_node(expression, pos, found.end(), Vec::new());
                        return Ok(Some(idx));
                    }
                }
                Ok(None)
            }
            ExpressionKind::Sequence(members) => {
                let mut new_pos = pos;
                let mut children = Vec::with_capacity(members.len());
                for member in members {
                    let child = self.match_core(member, new_pos)?;
                    let Some(child) = child else {
                        return Ok(None);
                    };
                    new_pos = self.node_end(child);
                    children.push(child);
                }
                let idx = self.push_node(expression, pos, new_pos, children);
                Ok(Some(idx))
            }
            ExpressionKind::OneOf(members) => {
                for member in members {
                    let child = self.match_core(member, pos)?;
                    if let Some(child) = child {
                        let end = self.node_end(child);
                        let idx = self.push_node(expression, pos, end, vec![child]);
                        return Ok(Some(idx));
                    }
                }
                Ok(None)
            }
            ExpressionKind::Lookahead { member, negative } => {
                let child = self.match_core(member, pos)?;
                let matched = child.is_some();
                let ok = (matched && *negative == false) || (matched == false && *negative);
                if ok {
                    let idx = self.push_node(expression, pos, pos, Vec::new());
                    return Ok(Some(idx));
                }
                Ok(None)
            }
            ExpressionKind::Quantifier { member, min, max } => {
                let mut new_pos = pos;
                let mut children: Vec<usize> = Vec::new();
                while new_pos < self.len {
                    let under_max = max.map(|m| children.len() < m).unwrap_or(true);
                    if under_max == false {
                        break;
                    }
                    let child = self.match_core(member, new_pos)?;
                    let Some(child) = child else {
                        break;
                    };
                    let end = self.node_end(child);
                    let length = end.saturating_sub(new_pos);
                    children.push(child);
                    if children.len() >= *min && length == 0 {
                        break;
                    }
                    new_pos = end;
                }

                if children.len() >= *min {
                    let idx = self.push_node(expression, pos, new_pos, children);
                    return Ok(Some(idx));
                }
                Ok(None)
            }
        }
    }

    fn materialize(&self, idx: usize) -> Node {
        let native = &self.arena[idx];
        let mut children: Vec<Node> = Vec::with_capacity(native.children.len());
        for child_idx in &native.children {
            children.push(self.materialize(*child_idx));
        }
        Node {
            expression: native.expression.clone(),
            start: native.start,
            end: native.end,
            children,
        }
    }
}

impl Expression {
    /// Create a literal expression.
    ///
    /// :param literal: Literal text to match.
    /// :returns: A literal expression.
    pub fn literal<S: Into<String>>(literal: S) -> Self {
        Self::new("", ExpressionKind::Literal(literal.into()))
    }

    /// Create a regex expression.
    ///
    /// :param pattern: Regex pattern.
    /// :returns: A regex expression.
    /// :raises regex::Error: If the pattern is invalid.
    pub fn regex(pattern: &str) -> Result<Self, regex::Error> {
        let regex = Regex::new(pattern)?;
        Ok(Self::new("", ExpressionKind::Regex(regex)))
    }

    /// Create a sequence expression.
    ///
    /// :param members: Ordered member expressions.
    /// :returns: A sequence expression.
    pub fn sequence(members: Vec<Expression>) -> Self {
        Self::new("", ExpressionKind::Sequence(members))
    }

    /// Create an ordered-choice expression.
    ///
    /// :param members: Alternative member expressions.
    /// :returns: An ordered-choice expression.
    pub fn one_of(members: Vec<Expression>) -> Self {
        Self::new("", ExpressionKind::OneOf(members))
    }

    /// Create a positive lookahead expression.
    ///
    /// :param member: The lookahead member expression.
    /// :returns: A lookahead expression.
    pub fn lookahead(member: Expression) -> Self {
        Self::new(
            "",
            ExpressionKind::Lookahead {
                member,
                negative: false,
            },
        )
    }

    /// Create a negative lookahead expression.
    ///
    /// :param member: The lookahead member expression.
    /// :returns: A negative lookahead expression.
    pub fn not(member: Expression) -> Self {
        Self::new(
            "",
            ExpressionKind::Lookahead {
                member,
                negative: true,
            },
        )
    }

    /// Create a quantifier expression.
    ///
    /// :param member: Quantified member expression.
    /// :param min: Minimum repetitions.
    /// :param max: Optional maximum repetitions (`None` means infinity).
    /// :returns: A quantifier expression.
    pub fn quantifier(member: Expression, min: usize, max: Option<usize>) -> Self {
        Self::new("", ExpressionKind::Quantifier { member, min, max })
    }

    /// Create an optional expression (`?`).
    ///
    /// :param member: Quantified member expression.
    /// :returns: An optional expression.
    pub fn optional(member: Expression) -> Self {
        Self::quantifier(member, 0, Some(1))
    }

    /// Create a zero-or-more expression (`*`).
    ///
    /// :param member: Quantified member expression.
    /// :returns: A zero-or-more expression.
    pub fn zero_or_more(member: Expression) -> Self {
        Self::quantifier(member, 0, None)
    }

    /// Create a one-or-more expression (`+`).
    ///
    /// :param member: Quantified member expression.
    /// :returns: A one-or-more expression.
    pub fn one_or_more(member: Expression) -> Self {
        Self::quantifier(member, 1, None)
    }

    /// Return a cloned expression with a different name.
    ///
    /// :param name: Rule name.
    /// :returns: A renamed expression.
    pub fn with_name(self, name: &str) -> Self {
        Self::new(name, self.inner.kind.clone())
    }

    /// Return the rule name.
    ///
    /// :returns: Rule name (possibly empty).
    pub fn name(&self) -> &str {
        &self.inner.name
    }

    /// Parse a full input string.
    ///
    /// :param input: Input text.
    /// :returns: Parse tree root node.
    /// :raises ParseError: On parse failures.
    pub fn parse(&self, input: &str) -> Result<Node, ParseError> {
        let node = self.match_prefix(input, 0)?;
        if node.end < input.len() {
            return Err(ParseError::Incomplete {
                rule: self.rule_name(),
                position: node.end,
            });
        }
        Ok(node)
    }

    /// Match at a position without requiring EOF consumption.
    ///
    /// :param input: Input text.
    /// :param position: Start byte offset.
    /// :returns: Parse tree node.
    /// :raises ParseError: On match failures.
    pub fn match_prefix(&self, input: &str, position: usize) -> Result<Node, ParseError> {
        if position > input.len() {
            return Err(ParseError::InvalidPosition {
                position,
                len: input.len(),
            });
        }
        let mut parser = Parser::new(input);
        let matched = parser.match_core(self, position)?;
        let Some(idx) = matched else {
            return Err(parser.error.to_error(self));
        };
        Ok(parser.materialize(idx))
    }

    /// Return the end offset of a successful full parse.
    ///
    /// :param input: Input text.
    /// :returns: End byte offset (`input.len()` on success).
    /// :raises ParseError: On parse failures.
    pub fn parse_end(&self, input: &str) -> Result<usize, ParseError> {
        let node = self.parse(input)?;
        Ok(node.end)
    }

    /// Return the end offset of a prefix match.
    ///
    /// :param input: Input text.
    /// :param position: Start byte offset.
    /// :returns: End offset if matched, else `None`.
    pub fn match_end(&self, input: &str, position: usize) -> Option<usize> {
        if position > input.len() {
            return None;
        }
        let mut parser = Parser::new(input);
        match parser.match_core(self, position) {
            Ok(Some(idx)) => Some(parser.node_end(idx)),
            Ok(None) => None,
            Err(_err) => None,
        }
    }

    /// Render this expression similarly to Parsimonious `as_rule()`.
    ///
    /// :returns: String representation.
    pub fn as_rule(&self) -> String {
        let rhs = self.as_rhs();
        if self.name().is_empty() {
            return rhs;
        }
        format!("{} = {}", self.name(), rhs)
    }

    fn new(name: &str, kind: ExpressionKind) -> Self {
        Self {
            inner: Arc::new(ExpressionInner {
                name: name.to_string(),
                kind,
            }),
        }
    }

    fn id(&self) -> usize {
        Arc::as_ptr(&self.inner) as usize
    }

    fn kind(&self) -> &ExpressionKind {
        &self.inner.kind
    }

    fn rule_name(&self) -> String {
        if self.name().is_empty() {
            return self.as_rhs();
        }
        self.name().to_string()
    }

    fn as_rhs(&self) -> String {
        match self.kind() {
            ExpressionKind::Literal(literal) => format!("{literal:?}"),
            ExpressionKind::Regex(regex) => format!("~{:?}", regex.as_str()),
            ExpressionKind::Sequence(members) => {
                let parts: Vec<String> = members.iter().map(Self::member_repr).collect();
                format!("({})", parts.join(" "))
            }
            ExpressionKind::OneOf(members) => {
                let parts: Vec<String> = members.iter().map(Self::member_repr).collect();
                format!("({})", parts.join(" / "))
            }
            ExpressionKind::Lookahead { member, negative } => {
                let op = if *negative { "!" } else { "&" };
                format!("{op}{}", Self::member_repr(member))
            }
            ExpressionKind::Quantifier { member, min, max } => {
                let qualifier = match (*min, *max) {
                    (0, Some(1)) => "?".to_string(),
                    (0, None) => "*".to_string(),
                    (1, None) => "+".to_string(),
                    (min, None) => format!("{{{min},}}"),
                    (0, Some(max)) => format!("{{,{max}}}"),
                    (min, Some(max)) => format!("{{{min},{max}}}"),
                };
                format!("{}{}", Self::member_repr(member), qualifier)
            }
        }
    }

    fn member_repr(member: &Expression) -> String {
        if member.name().is_empty() {
            return member.as_rhs();
        }
        member.name().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::{Expression, ParseError};

    #[test]
    fn literal_parse_success() {
        let expr = Expression::literal("hello").with_name("greeting");
        let node = expr.parse("hello").expect("parse should succeed");
        assert_eq!(node.start, 0);
        assert_eq!(node.end, 5);
    }

    #[test]
    fn sequence_and_choice_parse_success() {
        let hi = Expression::literal("hi");
        let hey = Expression::literal("hey");
        let space = Expression::literal(" ");
        let world = Expression::literal("world");

        let greeting = Expression::one_of(vec![hi, hey]);
        let phrase = Expression::sequence(vec![greeting, space, world]).with_name("phrase");

        let node = phrase.parse("hey world").expect("parse should succeed");
        assert_eq!(node.end, 9);
        assert_eq!(node.children.len(), 3);
    }

    #[test]
    fn lookahead_and_quantifier_parse_success() {
        let starts_with_a = Expression::sequence(vec![
            Expression::lookahead(Expression::literal("a")),
            Expression::one_or_more(Expression::regex("[a-z]").expect("regex must compile")),
        ]);

        let node = starts_with_a.parse("abc").expect("parse should succeed");
        assert_eq!(node.end, 3);
    }

    #[test]
    fn parse_incomplete_error() {
        let expr = Expression::literal("a");
        let err = expr.parse("ab").expect_err("parse should fail");
        assert_eq!(
            err,
            ParseError::Incomplete {
                rule: "\"a\"".to_string(),
                position: 1,
            }
        );
    }

    #[test]
    fn match_end_reports_prefix() {
        let expr = Expression::sequence(vec![
            Expression::literal("a"),
            Expression::optional(Expression::literal("b")),
        ]);
        assert_eq!(expr.match_end("ab", 0), Some(2));
        assert_eq!(expr.match_end("a", 0), Some(1));
        assert_eq!(expr.match_end("z", 0), None);
    }
}
