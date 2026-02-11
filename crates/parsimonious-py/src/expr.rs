use std::collections::HashSet;

use pyo3::basic::CompareOp;
use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyFloat, PyList, PyTuple, PyType};

use crate::cache::{CacheSlot, PackratCache};
use crate::node::{Node, RegexNode};

#[pyclass(subclass, dict)]
pub(crate) struct Expression {
    #[pyo3(get, set)]
    pub(crate) name: String,
}

#[pyclass(extends = Expression, subclass)]
pub(crate) struct Literal {
    #[pyo3(get)]
    pub(crate) literal: Py<PyAny>,
}

#[pyclass(extends = Expression, subclass)]
pub(crate) struct TokenMatcher {
    #[pyo3(get)]
    pub(crate) literal: Py<PyAny>,
}

#[pyclass(extends = Expression, subclass)]
pub(crate) struct Regex {
    #[pyo3(get)]
    pub(crate) re: Py<PyAny>,
}

#[pyclass(extends = Expression, subclass)]
pub(crate) struct Sequence {
    #[pyo3(get, set)]
    pub(crate) members: Py<PyTuple>,
}

#[pyclass(extends = Expression, subclass)]
pub(crate) struct OneOf {
    #[pyo3(get, set)]
    pub(crate) members: Py<PyTuple>,
}

#[pyclass(extends = Expression, subclass)]
pub(crate) struct Lookahead {
    #[pyo3(get, set)]
    pub(crate) members: Py<PyTuple>,
    #[pyo3(get)]
    pub(crate) negativity: bool,
}

#[pyclass(extends = Expression, subclass)]
pub(crate) struct Quantifier {
    #[pyo3(get, set)]
    pub(crate) members: Py<PyTuple>,
    #[pyo3(get)]
    pub(crate) min: usize,
    pub(crate) max: Option<usize>, // None => infinity
}

#[pyclass(extends = Expression, subclass)]
pub(crate) struct CustomExpr {
    pub(crate) callable: Py<PyAny>,
    pub(crate) arity: usize,
    pub(crate) grammar: Py<PyAny>,
    pub(crate) callable_name: String,
}

fn py_repr(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(obj.repr()?.to_str()?.to_string())
}

fn regex_flags_from_bits(bits: u32) -> String {
    // Mirrors parsimonious.expressions.Regex._regex_flags_from_bits.
    let flags = "ilmsuxa".as_bytes();
    let mut out = String::new();
    for i in 1..=flags.len() {
        if (bits & (1_u32 << i)) != 0 {
            out.push(flags[i - 1] as char);
        }
    }
    out
}

fn expr_name(expr_any: &Bound<'_, PyAny>) -> PyResult<String> {
    expr_any.getattr("name")?.extract()
}

fn unicode_members(py: Python<'_>, members: &Bound<'_, PyTuple>) -> PyResult<Vec<String>> {
    let mut out: Vec<String> = Vec::with_capacity(members.len());
    for m in members.iter() {
        let name: String = expr_name(&m)?;
        if name.len() > 0 {
            out.push(name);
        } else {
            out.push(as_rhs(py, &m)?);
        }
    }
    Ok(out)
}

fn as_rhs(py: Python<'_>, expr_any: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(tok) = expr_any.cast::<TokenMatcher>() {
        let tok_ref = tok.borrow();
        let repr = py_repr(tok_ref.literal.bind(py))?;
        return Ok(repr);
    }

    if let Ok(lit) = expr_any.cast::<Literal>() {
        let lit_ref = lit.borrow();
        let repr = py_repr(lit_ref.literal.bind(py))?;
        return Ok(repr);
    }

    if let Ok(regex) = expr_any.cast::<Regex>() {
        let regex_ref = regex.borrow();
        let re_any = regex_ref.re.bind(py);
        let pattern = re_any.getattr("pattern")?;
        let pattern_repr = py_repr(&pattern)?;
        let bits: u32 = re_any.getattr("flags")?.extract()?;
        let flags = regex_flags_from_bits(bits);
        return Ok(format!("~{pattern_repr}{flags}"));
    }

    if let Ok(seq) = expr_any.cast::<Sequence>() {
        let seq_ref = seq.borrow();
        let members = seq_ref.members.bind(py);
        let m = unicode_members(py, members)?;
        return Ok(format!("({})", m.join(" ")));
    }

    if let Ok(oneof) = expr_any.cast::<OneOf>() {
        let oneof_ref = oneof.borrow();
        let members = oneof_ref.members.bind(py);
        let m = unicode_members(py, members)?;
        return Ok(format!("({})", m.join(" / ")));
    }

    if let Ok(lookahead) = expr_any.cast::<Lookahead>() {
        let look_ref = lookahead.borrow();
        let members = look_ref.members.bind(py);
        let m = unicode_members(py, members)?;
        let op = if look_ref.negativity { "!" } else { "&" };
        if m.is_empty() {
            return Ok(format!("{op}"));
        }
        return Ok(format!("{op}{}", m[0]));
    }

    if let Ok(q) = expr_any.cast::<Quantifier>() {
        let q_ref = q.borrow();
        let members = q_ref.members.bind(py);
        let m = unicode_members(py, members)?;
        let qualifier = match (q_ref.min, q_ref.max) {
            (0, Some(1)) => "?".to_string(),
            (0, None) => "*".to_string(),
            (1, None) => "+".to_string(),
            (min, None) => format!("{{{min},}}"),
            (0, Some(max)) => format!("{{,{max}}}"),
            (min, Some(max)) => format!("{{{min},{max}}}"),
        };
        if m.is_empty() {
            return Ok(qualifier);
        }
        return Ok(format!("{}{}", m[0], qualifier));
    }

    if let Ok(custom) = expr_any.cast::<CustomExpr>() {
        let c = custom.borrow();
        return Ok(format!("{{custom function \"{}\"}}", c.callable_name));
    }

    // Support LazyReference / other compatibility objects during grammar compilation.
    if expr_any.hasattr("_as_rhs")? {
        let rhs = expr_any.call_method0("_as_rhs")?;
        return rhs.extract();
    }

    Err(PyRuntimeError::new_err(
        "Unknown expression type; cannot format as rule RHS.",
    ))
}

fn as_rule(py: Python<'_>, expr_any: &Bound<'_, PyAny>) -> PyResult<String> {
    let mut rhs = as_rhs(py, expr_any)?;
    if rhs.starts_with('(') && rhs.ends_with(')') && rhs.len() >= 2 {
        rhs = rhs[1..rhs.len() - 1].to_string();
    }
    let name = expr_name(expr_any)?;
    if name.len() > 0 {
        Ok(format!("{name} = {rhs}"))
    } else {
        Ok(rhs)
    }
}

fn parse_error_type(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    Ok(PyModule::import(py, "parsimonious.exceptions")?
        .getattr("ParseError")?
        .into_any())
}

fn incomplete_parse_error_type(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    Ok(PyModule::import(py, "parsimonious.exceptions")?
        .getattr("IncompleteParseError")?
        .into_any())
}

fn left_recursion_error_type(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    Ok(PyModule::import(py, "parsimonious.exceptions")?
        .getattr("LeftRecursionError")?
        .into_any())
}

fn update_error(
    _py: Python<'_>,
    expr_any: &Bound<'_, PyAny>,
    pos: usize,
    error: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let error_pos: isize = error.getattr("pos")?.extract()?;
    if (pos as isize) < error_pos {
        return Ok(());
    }

    let self_name = expr_name(expr_any)?;
    let existing_expr = error.getattr("expr")?;
    let existing_is_none = existing_expr.is_none();
    if self_name.len() == 0 && existing_is_none == false {
        return Ok(());
    }

    error.setattr("expr", expr_any)?;
    error.setattr("pos", pos)?;
    Ok(())
}

fn new_node(
    py: Python<'_>,
    expr: &Bound<'_, PyAny>,
    text: &Bound<'_, PyAny>,
    start: usize,
    end: usize,
    children: &[Py<PyAny>],
) -> PyResult<Py<PyAny>> {
    let children_list = PyList::new(py, children)?;
    let node = Bound::new(
        py,
        Node {
            expr: expr.clone().unbind(),
            full_text: text.clone().unbind(),
            start,
            end,
            children: children_list.unbind(),
        },
    )?;
    Ok(node.into_any().unbind())
}

fn new_regex_node(
    py: Python<'_>,
    expr: &Bound<'_, PyAny>,
    text: &Bound<'_, PyAny>,
    start: usize,
    end: usize,
    match_obj: Py<PyAny>,
) -> PyResult<Py<PyAny>> {
    let base = Node {
        expr: expr.clone().unbind(),
        full_text: text.clone().unbind(),
        start,
        end,
        children: PyList::empty(py).unbind(),
    };
    let init = PyClassInitializer::from(base).add_subclass(RegexNode { r#match: match_obj });
    let node: Bound<'_, RegexNode> = Bound::new(py, init)?;
    Ok(node.into_any().unbind())
}

fn node_end(py: Python<'_>, node: &Py<PyAny>) -> PyResult<usize> {
    let bound = node.bind(py);
    let as_node: &Bound<'_, Node> = bound.cast()?;
    Ok(as_node.borrow().end)
}

fn match_core_any(
    py: Python<'_>,
    expr_any: &Bound<'_, PyAny>,
    text: &Bound<'_, PyAny>,
    pos: usize,
    cache: &Bound<'_, PackratCache>,
    error: &Bound<'_, PyAny>,
) -> PyResult<Option<Py<PyAny>>> {
    let expr_id: usize = expr_any.as_ptr() as usize;

    let cache_len: usize = { cache.borrow().len };
    if pos > cache_len {
        return Ok(None);
    }

    {
        let mut cache_ref = cache.borrow_mut();
        let slots = cache_ref.slots_for_expr_mut(expr_id);
        match &slots[pos] {
            CacheSlot::Unknown => {
                slots[pos] = CacheSlot::InProgress;
            }
            CacheSlot::InProgress => {
                let ty = left_recursion_error_type(py)?;
                let inst = ty.call1((text, -1isize, expr_any))?;
                return Err(PyErr::from_value(inst));
            }
            CacheSlot::Done(obj) => {
                if obj.bind(py).is_none() {
                    return Ok(None);
                }
                return Ok(Some(obj.clone_ref(py)));
            }
        }
    }

    let result = uncached_match(py, expr_any, text, pos, cache, error)?;

    {
        let mut cache_ref = cache.borrow_mut();
        let slots = cache_ref.slots_for_expr_mut(expr_id);
        let stored = match &result {
            Some(obj) => obj.clone_ref(py),
            None => py.None(),
        };
        slots[pos] = CacheSlot::Done(stored);
    }

    if result.is_none() {
        update_error(py, expr_any, pos, error)?;
    }

    Ok(result)
}

fn uncached_match(
    py: Python<'_>,
    expr_any: &Bound<'_, PyAny>,
    text: &Bound<'_, PyAny>,
    pos: usize,
    cache: &Bound<'_, PackratCache>,
    error: &Bound<'_, PyAny>,
) -> PyResult<Option<Py<PyAny>>> {
    if let Ok(tok) = expr_any.cast::<TokenMatcher>() {
        let tok_ref = tok.borrow();
        let size: usize = { cache.borrow().len };
        if pos >= size {
            return Ok(None);
        }

        let item = text.get_item(pos)?;
        let ty = item.getattr("type")?;
        let ok: bool = ty.eq(tok_ref.literal.bind(py))?;
        if ok {
            let end = pos + 1;
            let node = new_node(py, expr_any, text, pos, end, &[])?;
            return Ok(Some(node));
        }
        return Ok(None);
    }

    if let Ok(lit) = expr_any.cast::<Literal>() {
        let lit_ref = lit.borrow();
        let literal = lit_ref.literal.bind(py);
        let ok: bool = text.call_method1("startswith", (literal, pos))?.extract()?;
        if ok {
            let lit_len = literal.len()?;
            let end = pos + lit_len;
            let node = new_node(py, expr_any, text, pos, end, &[])?;
            return Ok(Some(node));
        }
        return Ok(None);
    }

    if let Ok(regex) = expr_any.cast::<Regex>() {
        let regex_ref = regex.borrow();
        let re_any = regex_ref.re.bind(py);
        let m = re_any.call_method1("match", (text, pos))?;
        if m.is_none() {
            return Ok(None);
        }
        let end: usize = m.call_method0("end")?.extract()?;
        let node = new_regex_node(py, expr_any, text, pos, end, m.unbind())?;
        return Ok(Some(node));
    }

    if let Ok(seq) = expr_any.cast::<Sequence>() {
        let seq_ref = seq.borrow();
        let members = seq_ref.members.bind(py);

        let mut new_pos = pos;
        let mut children: Vec<Py<PyAny>> = Vec::with_capacity(members.len());
        for m in members.iter() {
            let node = match_core_any(py, &m, text, new_pos, cache, error)?;
            let Some(node) = node else {
                return Ok(None);
            };
            new_pos = node_end(py, &node)?;
            children.push(node);
        }

        let node = new_node(py, expr_any, text, pos, new_pos, &children)?;
        return Ok(Some(node));
    }

    if let Ok(oneof) = expr_any.cast::<OneOf>() {
        let oneof_ref = oneof.borrow();
        let members = oneof_ref.members.bind(py);
        for m in members.iter() {
            let node = match_core_any(py, &m, text, pos, cache, error)?;
            if let Some(child) = node {
                let end = node_end(py, &child)?;
                let node = new_node(py, expr_any, text, pos, end, &[child])?;
                return Ok(Some(node));
            }
        }
        return Ok(None);
    }

    if let Ok(lookahead) = expr_any.cast::<Lookahead>() {
        let look_ref = lookahead.borrow();
        let members = look_ref.members.bind(py);
        if members.len() == 0 {
            return Ok(None);
        }
        let member0 = members.get_item(0)?;
        let node = match_core_any(py, &member0, text, pos, cache, error)?;
        let ok = (node.is_none()) == look_ref.negativity;
        if ok {
            let node = new_node(py, expr_any, text, pos, pos, &[])?;
            return Ok(Some(node));
        }
        return Ok(None);
    }

    if let Ok(q) = expr_any.cast::<Quantifier>() {
        let q_ref = q.borrow();
        let members = q_ref.members.bind(py);
        if members.len() == 0 {
            return Ok(None);
        }

        let member0 = members.get_item(0)?;
        let mut new_pos = pos;
        let mut children: Vec<Py<PyAny>> = Vec::new();

        let size: usize = { cache.borrow().len };
        while new_pos < size && (q_ref.max.is_none() || children.len() < q_ref.max.unwrap()) {
            let node = match_core_any(py, &member0, text, new_pos, cache, error)?;
            let Some(child) = node else {
                break;
            };
            let child_end = node_end(py, &child)?;
            let length = child_end.saturating_sub(new_pos);
            children.push(child);
            if children.len() >= q_ref.min && length == 0 {
                break;
            }
            new_pos = child_end;
        }

        if children.len() >= q_ref.min {
            let node = new_node(py, expr_any, text, pos, new_pos, &children)?;
            return Ok(Some(node));
        }
        return Ok(None);
    }

    if let Ok(custom) = expr_any.cast::<CustomExpr>() {
        let custom_ref = custom.borrow();
        let result = if custom_ref.arity == 2 {
            custom_ref.callable.bind(py).call1((text, pos))?
        } else {
            let cache_obj: Py<PackratCache> = cache.clone().unbind();
            let error_obj: Py<PyAny> = error.clone().unbind();
            let grammar_obj: Py<PyAny> = custom_ref.grammar.clone_ref(py);
            custom_ref
                .callable
                .bind(py)
                .call1((text, pos, cache_obj, error_obj, grammar_obj))?
        };

        if result.is_none() {
            return Ok(None);
        }

        if let Ok(end) = result.extract::<usize>() {
            let node = new_node(py, expr_any, text, pos, end, &[])?;
            return Ok(Some(node));
        }

        if let Ok(tup) = result.cast::<PyTuple>() {
            if tup.len() != 2 {
                return Err(PyTypeError::new_err(
                    "Custom rule must return an int, (end, children), Node, or None.",
                ));
            }
            let end: usize = tup.get_item(0)?.extract()?;
            let children_any = tup.get_item(1)?;
            if children_any.is_none() {
                let node = new_node(py, expr_any, text, pos, end, &[])?;
                return Ok(Some(node));
            }
            let children_list = children_any.try_iter()?;
            let mut children: Vec<Py<PyAny>> = Vec::new();
            for c in children_list {
                children.push(c?.unbind());
            }
            let node = new_node(py, expr_any, text, pos, end, &children)?;
            return Ok(Some(node));
        }

        // Node-like or other: pass through (matches the original behavior).
        Ok(Some(result.unbind()))
    } else {
        Err(PyRuntimeError::new_err(
            "Unknown expression type; cannot match.",
        ))
    }
}

fn eq_expr(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    checked: &mut HashSet<(usize, usize)>,
) -> PyResult<bool> {
    if a.get_type().is(b.get_type()) == false {
        return Ok(false);
    }

    let key = (a.as_ptr() as usize, b.as_ptr() as usize);
    if checked.contains(&key) {
        return Ok(true);
    }
    checked.insert(key);

    if let Ok(la) = a.cast::<Literal>() {
        let lb: &Bound<'_, Literal> = b.cast()?;
        let na: String = a.getattr("name")?.extract()?;
        let nb: String = b.getattr("name")?.extract()?;
        if na != nb {
            return Ok(false);
        }
        let la_ref = la.borrow();
        let lb_ref = lb.borrow();
        return la_ref.literal.bind(py).eq(lb_ref.literal.bind(py));
    }

    if let Ok(ta) = a.cast::<TokenMatcher>() {
        let tb: &Bound<'_, TokenMatcher> = b.cast()?;
        let na: String = a.getattr("name")?.extract()?;
        let nb: String = b.getattr("name")?.extract()?;
        if na != nb {
            return Ok(false);
        }
        let ta_ref = ta.borrow();
        let tb_ref = tb.borrow();
        return ta_ref.literal.bind(py).eq(tb_ref.literal.bind(py));
    }

    if let Ok(ra) = a.cast::<Regex>() {
        let rb: &Bound<'_, Regex> = b.cast()?;
        let na: String = a.getattr("name")?.extract()?;
        let nb: String = b.getattr("name")?.extract()?;
        if na != nb {
            return Ok(false);
        }
        let ra_ref = ra.borrow();
        let rb_ref = rb.borrow();
        let pa = ra_ref.re.bind(py);
        let pb = rb_ref.re.bind(py);
        let pa_pat = pa.getattr("pattern")?;
        let pb_pat = pb.getattr("pattern")?;
        if pa_pat.eq(&pb_pat)? == false {
            return Ok(false);
        }
        let fa: u32 = pa.getattr("flags")?.extract()?;
        let fb: u32 = pb.getattr("flags")?.extract()?;
        return Ok(fa == fb);
    }

    if let Ok(sa) = a.cast::<Sequence>() {
        let sb: &Bound<'_, Sequence> = b.cast()?;
        let na: String = a.getattr("name")?.extract()?;
        let nb: String = b.getattr("name")?.extract()?;
        if na != nb {
            return Ok(false);
        }
        let sa_ref = sa.borrow();
        let sb_ref = sb.borrow();
        let ma = sa_ref.members.bind(py);
        let mb = sb_ref.members.bind(py);
        if ma.len() != mb.len() {
            return Ok(false);
        }
        for (ea, eb) in ma.iter().zip(mb.iter()) {
            if eq_expr(py, &ea, &eb, checked)? == false {
                return Ok(false);
            }
        }
        return Ok(true);
    }

    if let Ok(oa) = a.cast::<OneOf>() {
        let ob: &Bound<'_, OneOf> = b.cast()?;
        let na: String = a.getattr("name")?.extract()?;
        let nb: String = b.getattr("name")?.extract()?;
        if na != nb {
            return Ok(false);
        }
        let oa_ref = oa.borrow();
        let ob_ref = ob.borrow();
        let ma = oa_ref.members.bind(py);
        let mb = ob_ref.members.bind(py);
        if ma.len() != mb.len() {
            return Ok(false);
        }
        for (ea, eb) in ma.iter().zip(mb.iter()) {
            if eq_expr(py, &ea, &eb, checked)? == false {
                return Ok(false);
            }
        }
        return Ok(true);
    }

    if let Ok(la) = a.cast::<Lookahead>() {
        let lb: &Bound<'_, Lookahead> = b.cast()?;
        let na: String = a.getattr("name")?.extract()?;
        let nb: String = b.getattr("name")?.extract()?;
        if na != nb {
            return Ok(false);
        }
        let la_ref = la.borrow();
        let lb_ref = lb.borrow();
        if la_ref.negativity != lb_ref.negativity {
            return Ok(false);
        }
        let ma = la_ref.members.bind(py);
        let mb = lb_ref.members.bind(py);
        if ma.len() != mb.len() {
            return Ok(false);
        }
        if ma.len() == 0 {
            return Ok(true);
        }
        return eq_expr(py, &ma.get_item(0)?, &mb.get_item(0)?, checked);
    }

    if let Ok(qa) = a.cast::<Quantifier>() {
        let qb: &Bound<'_, Quantifier> = b.cast()?;
        let na: String = a.getattr("name")?.extract()?;
        let nb: String = b.getattr("name")?.extract()?;
        if na != nb {
            return Ok(false);
        }
        let qa_ref = qa.borrow();
        let qb_ref = qb.borrow();
        if qa_ref.min != qb_ref.min || qa_ref.max != qb_ref.max {
            return Ok(false);
        }
        let ma = qa_ref.members.bind(py);
        let mb = qb_ref.members.bind(py);
        if ma.len() != mb.len() {
            return Ok(false);
        }
        if ma.len() == 0 {
            return Ok(true);
        }
        return eq_expr(py, &ma.get_item(0)?, &mb.get_item(0)?, checked);
    }

    if let Ok(ca) = a.cast::<CustomExpr>() {
        let cb: &Bound<'_, CustomExpr> = b.cast()?;
        let na: String = a.getattr("name")?.extract()?;
        let nb: String = b.getattr("name")?.extract()?;
        if na != nb {
            return Ok(false);
        }
        let ca_ref = ca.borrow();
        let cb_ref = cb.borrow();
        if ca_ref.arity != cb_ref.arity {
            return Ok(false);
        }
        let callables_eq = ca_ref.callable.bind(py).eq(cb_ref.callable.bind(py))?;
        let grammars_eq = ca_ref.grammar.bind(py).eq(cb_ref.grammar.bind(py))?;
        return Ok(callables_eq && grammars_eq);
    }

    // Fall back to object identity when unsure.
    Ok(a.as_ptr() == b.as_ptr())
}

#[pymethods]
impl Expression {
    #[new]
    #[pyo3(signature = (name=""))]
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
        }
    }

    fn __hash__(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<isize> {
        // Mirror Expression.identity_tuple = (self.name,)
        let self_any: Bound<'_, PyAny> = unsafe { Bound::from_borrowed_ptr(py, slf.as_ptr()) };
        let name: String = self_any.getattr("name")?.extract()?;
        let tup = PyTuple::new(py, [name])?;
        tup.as_any().hash()
    }

    fn __richcmp__(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        op: CompareOp,
    ) -> PyResult<Py<PyAny>> {
        let self_any: Bound<'_, PyAny> = unsafe { Bound::from_borrowed_ptr(py, slf.as_ptr()) };

        match op {
            CompareOp::Eq => {
                let mut checked: HashSet<(usize, usize)> = HashSet::new();
                let eq = eq_expr(py, &self_any, other, &mut checked).unwrap_or(false);
                Ok(eq.into_pyobject(py)?.to_owned().into_any().unbind())
            }
            CompareOp::Ne => {
                let mut checked: HashSet<(usize, usize)> = HashSet::new();
                let eq = eq_expr(py, &self_any, other, &mut checked).unwrap_or(false);
                Ok((!eq).into_pyobject(py)?.to_owned().into_any().unbind())
            }
            _ => Ok(py.NotImplemented()),
        }
    }

    fn as_rule(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<String> {
        let self_any: Bound<'_, PyAny> = unsafe { Bound::from_borrowed_ptr(py, slf.as_ptr()) };
        as_rule(py, &self_any)
    }

    fn __str__(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<String> {
        let self_any: Bound<'_, PyAny> = unsafe { Bound::from_borrowed_ptr(py, slf.as_ptr()) };
        let class_name = self_any.get_type().name()?.to_string();
        let rule = as_rule(py, &self_any)?;
        Ok(format!("<{class_name} {rule}>"))
    }

    fn __repr__(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<String> {
        Expression::__str__(slf, py)
    }

    #[pyo3(signature = (text, pos=0))]
    fn r#match(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        text: Py<PyAny>,
        pos: usize,
    ) -> PyResult<Py<PyAny>> {
        let cache: Bound<'_, PackratCache> =
            Bound::new(py, PackratCache::from_text(py, text.clone_ref(py))?)?;
        let parse_error = parse_error_type(py)?;
        let error = parse_error.call1((text.bind(py),))?;

        let self_any: Bound<'_, PyAny> = unsafe { Bound::from_borrowed_ptr(py, slf.as_ptr()) };
        let node = match_core_any(py, &self_any, text.bind(py), pos, &cache, &error)?;
        let Some(node) = node else {
            return Err(PyErr::from_value(error));
        };
        Ok(node)
    }

    #[pyo3(signature = (text, pos, cache, error))]
    fn match_core(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        text: Py<PyAny>,
        pos: usize,
        cache: Py<PackratCache>,
        error: Py<PyAny>,
    ) -> PyResult<Option<Py<PyAny>>> {
        let self_any: Bound<'_, PyAny> = unsafe { Bound::from_borrowed_ptr(py, slf.as_ptr()) };
        match_core_any(
            py,
            &self_any,
            text.bind(py),
            pos,
            cache.bind(py),
            error.bind(py),
        )
    }

    #[pyo3(signature = (text, pos=0))]
    fn parse(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        text: Py<PyAny>,
        pos: usize,
    ) -> PyResult<Py<PyAny>> {
        let self_any: Bound<'_, PyAny> = unsafe { Bound::from_borrowed_ptr(py, slf.as_ptr()) };
        let node = Expression::r#match(slf, py, text.clone_ref(py), pos)?;
        let end = node_end(py, &node)?;

        let size = match text.bind(py).len() {
            Ok(l) => l,
            Err(_err) => {
                return Err(PyTypeError::new_err("Unsupported input type."));
            }
        };

        if end < size {
            let ty = incomplete_parse_error_type(py)?;
            let inst = ty.call1((text.bind(py), end, self_any))?;
            return Err(PyErr::from_value(inst));
        }
        Ok(node)
    }

    fn resolve_refs(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        _rule_map: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let self_any: Bound<'_, PyAny> = unsafe { Bound::from_borrowed_ptr(py, slf.as_ptr()) };
        Ok(self_any.unbind())
    }

    #[classmethod]
    fn custom(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        rule_name: String,
        callable: Py<PyAny>,
        arity: usize,
        grammar: Py<PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let callable_name: String = callable
            .bind(py)
            .getattr("__name__")
            .and_then(|v| v.extract::<String>())
            .unwrap_or_else(|_err| "<callable>".to_string());

        let base = Expression { name: rule_name };
        let init = PyClassInitializer::from(base).add_subclass(CustomExpr {
            callable,
            arity,
            grammar,
            callable_name,
        });
        let obj: Bound<'_, CustomExpr> = Bound::new(py, init)?;
        Ok(obj.into_any().unbind())
    }
}

#[pymethods]
impl Literal {
    #[new]
    #[pyo3(signature = (literal, name=""))]
    fn new(py: Python<'_>, literal: Py<PyAny>, name: &str) -> PyResult<PyClassInitializer<Self>> {
        let base = Expression {
            name: name.to_string(),
        };
        Ok(PyClassInitializer::from(base).add_subclass(Self { literal }))
    }

    fn __hash__(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<isize> {
        // Mirror Literal.identity_tuple = (name, literal)
        let self_any: Bound<'_, PyAny> = unsafe { Bound::from_borrowed_ptr(py, slf.as_ptr()) };
        let name: String = self_any.getattr("name")?.extract()?;
        let literal = self_any.getattr("literal")?;
        let tup = PyTuple::new(py, [name.into_pyobject(py)?.into_any(), literal])?;
        tup.as_any().hash()
    }
}

#[pymethods]
impl TokenMatcher {
    #[new]
    #[pyo3(signature = (literal, name=""))]
    fn new(py: Python<'_>, literal: Py<PyAny>, name: &str) -> PyResult<PyClassInitializer<Self>> {
        let base = Expression {
            name: name.to_string(),
        };
        Ok(PyClassInitializer::from(base).add_subclass(Self { literal }))
    }

    fn __hash__(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<isize> {
        // Same identity tuple as Literal: (name, literal)
        let self_any: Bound<'_, PyAny> = unsafe { Bound::from_borrowed_ptr(py, slf.as_ptr()) };
        let name: String = self_any.getattr("name")?.extract()?;
        let literal = self_any.getattr("literal")?;
        let tup = PyTuple::new(py, [name.into_pyobject(py)?.into_any(), literal])?;
        tup.as_any().hash()
    }
}

#[pymethods]
impl Regex {
    #[new]
    #[pyo3(signature = (pattern, name="", ignore_case=false, locale=false, multiline=false, dot_all=false, unicode=false, verbose=false, ascii=false))]
    fn new(
        py: Python<'_>,
        pattern: Py<PyAny>,
        name: &str,
        ignore_case: bool,
        locale: bool,
        multiline: bool,
        dot_all: bool,
        unicode: bool,
        verbose: bool,
        ascii: bool,
    ) -> PyResult<PyClassInitializer<Self>> {
        let re_mod = PyModule::import(py, "regex").or_else(|_err| PyModule::import(py, "re"))?;
        let mut flags: u32 = 0;
        if ignore_case {
            flags |= re_mod.getattr("I")?.extract::<u32>()?;
        }
        if locale {
            flags |= re_mod.getattr("L")?.extract::<u32>()?;
        }
        if multiline {
            flags |= re_mod.getattr("M")?.extract::<u32>()?;
        }
        if dot_all {
            flags |= re_mod.getattr("S")?.extract::<u32>()?;
        }
        if unicode {
            flags |= re_mod.getattr("U")?.extract::<u32>()?;
        }
        if verbose {
            flags |= re_mod.getattr("X")?.extract::<u32>()?;
        }
        if ascii {
            flags |= re_mod.getattr("A")?.extract::<u32>()?;
        }

        let compiled = re_mod
            .getattr("compile")?
            .call1((pattern.bind(py), flags))?;

        let base = Expression {
            name: name.to_string(),
        };
        Ok(PyClassInitializer::from(base).add_subclass(Self {
            re: compiled.unbind(),
        }))
    }

    fn __hash__(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<isize> {
        // Mirror Regex.identity_tuple = (name, compiled_regex)
        let self_any: Bound<'_, PyAny> = unsafe { Bound::from_borrowed_ptr(py, slf.as_ptr()) };
        let name: String = self_any.getattr("name")?.extract()?;
        let re_obj = self_any.getattr("re")?;
        let tup = PyTuple::new(py, [name.into_pyobject(py)?.into_any(), re_obj])?;
        tup.as_any().hash()
    }
}

#[pymethods]
impl Sequence {
    #[new]
    #[pyo3(signature = (*members, name=""))]
    fn new(
        py: Python<'_>,
        members: Bound<'_, PyTuple>,
        name: &str,
    ) -> PyResult<PyClassInitializer<Self>> {
        let base = Expression {
            name: name.to_string(),
        };
        Ok(PyClassInitializer::from(base).add_subclass(Self {
            members: members.unbind(),
        }))
    }

    fn __hash__(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<isize> {
        // Mirror Compound.__hash__: hash((self.__class__, self.name))
        let self_any: Bound<'_, PyAny> = unsafe { Bound::from_borrowed_ptr(py, slf.as_ptr()) };
        let ty = self_any.get_type();
        let name: String = self_any.getattr("name")?.extract()?;
        let tup = PyTuple::new(py, [ty.into_any(), name.into_pyobject(py)?.into_any()])?;
        tup.as_any().hash()
    }

    fn resolve_refs(
        mut slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        rule_map: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let members = slf.members.bind(py);
        let mut new_members: Vec<Py<PyAny>> = Vec::with_capacity(members.len());
        for m in members.iter() {
            let resolved = if m.hasattr("resolve_refs")? {
                m.call_method1("resolve_refs", (rule_map,))?
            } else {
                m
            };
            new_members.push(resolved.unbind());
        }
        let new_tuple = PyTuple::new(py, new_members)?;
        slf.members = new_tuple.unbind();

        let self_any: Bound<'_, PyAny> = unsafe { Bound::from_borrowed_ptr(py, slf.as_ptr()) };
        Ok(self_any.unbind())
    }
}

#[pymethods]
impl OneOf {
    #[new]
    #[pyo3(signature = (*members, name=""))]
    fn new(
        py: Python<'_>,
        members: Bound<'_, PyTuple>,
        name: &str,
    ) -> PyResult<PyClassInitializer<Self>> {
        let base = Expression {
            name: name.to_string(),
        };
        Ok(PyClassInitializer::from(base).add_subclass(Self {
            members: members.unbind(),
        }))
    }

    fn __hash__(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<isize> {
        let self_any: Bound<'_, PyAny> = unsafe { Bound::from_borrowed_ptr(py, slf.as_ptr()) };
        let ty = self_any.get_type();
        let name: String = self_any.getattr("name")?.extract()?;
        let tup = PyTuple::new(py, [ty.into_any(), name.into_pyobject(py)?.into_any()])?;
        tup.as_any().hash()
    }

    fn resolve_refs(
        mut slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        rule_map: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let members = slf.members.bind(py);
        let mut new_members: Vec<Py<PyAny>> = Vec::with_capacity(members.len());
        for m in members.iter() {
            let resolved = if m.hasattr("resolve_refs")? {
                m.call_method1("resolve_refs", (rule_map,))?
            } else {
                m
            };
            new_members.push(resolved.unbind());
        }
        let new_tuple = PyTuple::new(py, new_members)?;
        slf.members = new_tuple.unbind();

        let self_any: Bound<'_, PyAny> = unsafe { Bound::from_borrowed_ptr(py, slf.as_ptr()) };
        Ok(self_any.unbind())
    }
}

#[pymethods]
impl Lookahead {
    #[new]
    #[pyo3(signature = (member, negative=false, name=""))]
    fn new(
        py: Python<'_>,
        member: Py<PyAny>,
        negative: bool,
        name: &str,
    ) -> PyResult<PyClassInitializer<Self>> {
        let base = Expression {
            name: name.to_string(),
        };
        let members = PyTuple::new(py, [member])?;
        Ok(PyClassInitializer::from(base).add_subclass(Self {
            members: members.unbind(),
            negativity: negative,
        }))
    }

    fn __hash__(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<isize> {
        let self_any: Bound<'_, PyAny> = unsafe { Bound::from_borrowed_ptr(py, slf.as_ptr()) };
        let ty = self_any.get_type();
        let name: String = self_any.getattr("name")?.extract()?;
        let tup = PyTuple::new(py, [ty.into_any(), name.into_pyobject(py)?.into_any()])?;
        tup.as_any().hash()
    }

    fn resolve_refs(
        mut slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        rule_map: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let members = slf.members.bind(py);
        let mut new_members: Vec<Py<PyAny>> = Vec::with_capacity(members.len());
        for m in members.iter() {
            let resolved = if m.hasattr("resolve_refs")? {
                m.call_method1("resolve_refs", (rule_map,))?
            } else {
                m
            };
            new_members.push(resolved.unbind());
        }
        let new_tuple = PyTuple::new(py, new_members)?;
        slf.members = new_tuple.unbind();

        let self_any: Bound<'_, PyAny> = unsafe { Bound::from_borrowed_ptr(py, slf.as_ptr()) };
        Ok(self_any.unbind())
    }
}

#[pymethods]
impl Quantifier {
    #[new]
    #[pyo3(signature = (member=None, min=0usize, max=None, name=""))]
    fn new(
        py: Python<'_>,
        member: Option<Py<PyAny>>,
        min: usize,
        max: Option<Py<PyAny>>,
        name: &str,
    ) -> PyResult<PyClassInitializer<Self>> {
        let base = Expression {
            name: name.to_string(),
        };

        let members = match member {
            Some(m) => PyTuple::new(py, [m])?,
            None => PyTuple::empty(py),
        };

        let max_parsed = match max {
            None => None,
            Some(m) => {
                let m_any = m.bind(py);
                if let Ok(f) = m_any.extract::<f64>() {
                    if f.is_infinite() && f.is_sign_positive() {
                        None
                    } else {
                        return Err(PyTypeError::new_err("max must be an int or infinity."));
                    }
                } else {
                    Some(m_any.extract::<usize>()?)
                }
            }
        };

        Ok(PyClassInitializer::from(base).add_subclass(Self {
            members: members.unbind(),
            min,
            max: max_parsed,
        }))
    }

    #[getter]
    fn max(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self.max {
            Some(m) => Ok(m.into_pyobject(py)?.into_any().unbind()),
            None => Ok(PyFloat::new(py, f64::INFINITY).into_any().unbind()),
        }
    }

    fn __hash__(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<isize> {
        let self_any: Bound<'_, PyAny> = unsafe { Bound::from_borrowed_ptr(py, slf.as_ptr()) };
        let ty = self_any.get_type();
        let name: String = self_any.getattr("name")?.extract()?;
        let tup = PyTuple::new(py, [ty.into_any(), name.into_pyobject(py)?.into_any()])?;
        tup.as_any().hash()
    }

    fn resolve_refs(
        mut slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        rule_map: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let members = slf.members.bind(py);
        let mut new_members: Vec<Py<PyAny>> = Vec::with_capacity(members.len());
        for m in members.iter() {
            let resolved = if m.hasattr("resolve_refs")? {
                m.call_method1("resolve_refs", (rule_map,))?
            } else {
                m
            };
            new_members.push(resolved.unbind());
        }
        let new_tuple = PyTuple::new(py, new_members)?;
        slf.members = new_tuple.unbind();

        let self_any: Bound<'_, PyAny> = unsafe { Bound::from_borrowed_ptr(py, slf.as_ptr()) };
        Ok(self_any.unbind())
    }
}

#[pymethods]
impl CustomExpr {
    fn __hash__(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<isize> {
        let self_any: Bound<'_, PyAny> = unsafe { Bound::from_borrowed_ptr(py, slf.as_ptr()) };
        let name: String = self_any.getattr("name")?.extract()?;
        let tup = PyTuple::new(py, [name])?;
        tup.as_any().hash()
    }
}
