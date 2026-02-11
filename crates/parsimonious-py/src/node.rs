use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyList, PySlice};

#[pyclass(subclass)]
pub(crate) struct Node {
    #[pyo3(get, set)]
    pub(crate) expr: Py<PyAny>,
    #[pyo3(get, set)]
    pub(crate) full_text: Py<PyAny>,
    #[pyo3(get, set)]
    pub(crate) start: usize,
    #[pyo3(get, set)]
    pub(crate) end: usize,
    #[pyo3(get, set)]
    pub(crate) children: Py<PyList>,
}

#[pyclass(extends = Node)]
pub(crate) struct RegexNode {
    #[pyo3(get, set)]
    pub(crate) r#match: Py<PyAny>,
}

#[pymethods]
impl Node {
    #[new]
    #[pyo3(signature = (expr, full_text, start, end, children=None))]
    fn new(
        py: Python<'_>,
        expr: Py<PyAny>,
        full_text: Py<PyAny>,
        start: usize,
        end: usize,
        children: Option<Py<PyList>>,
    ) -> PyResult<Self> {
        let children = match children {
            Some(c) => c,
            None => PyList::empty(py).unbind(),
        };
        Ok(Self {
            expr,
            full_text,
            start,
            end,
            children,
        })
    }

    #[getter]
    fn expr_name(&self, py: Python<'_>) -> PyResult<String> {
        let name_any = self.expr.bind(py).getattr("name")?;
        let name: String = name_any.extract()?;
        Ok(name)
    }

    #[getter]
    fn text(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let slice = PySlice::new(py, self.start as isize, self.end as isize, 1);
        let part = self.full_text.bind(py).get_item(slice)?;
        Ok(part.unbind())
    }

    fn __iter__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let iter = self.children.bind(py).as_any().try_iter()?;
        Ok(iter.into_any().unbind())
    }

    #[pyo3(signature = (error=None))]
    fn prettily(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        error: Option<Py<PyAny>>,
    ) -> PyResult<String> {
        fn indent(text: &str) -> String {
            text.split('\n')
                .map(|line| format!("    {line}"))
                .collect::<Vec<_>>()
                .join("\n")
        }

        let self_ptr = slf.as_ptr() as usize;
        let error_is_self = match &error {
            Some(err) => (err.bind(py).as_ptr() as usize) == self_ptr,
            None => false,
        };

        let class_name = unsafe { Bound::from_borrowed_ptr(py, slf.as_ptr()) }
            .into_any()
            .get_type()
            .name()?
            .to_string();
        let expr_name = slf.expr_name(py)?;

        let text_obj = slf.text(py)?;
        let text_str = text_obj.bind(py).str()?.to_str()?.to_string();

        let called = if expr_name.len() > 0 {
            format!(" called \"{expr_name}\"")
        } else {
            String::new()
        };

        let highlight = if error_is_self {
            "  <-- *** We were here. ***"
        } else {
            ""
        };

        let mut ret: Vec<String> = vec![format!(
            "<{class_name}{called} matching \"{text_str}\">{highlight}"
        )];

        for child_any in slf.children.bind(py).iter() {
            let error_arg: Option<Py<PyAny>> = match &error {
                Some(e) => Some(e.clone_ref(py)),
                None => None,
            };
            let pretty: String = child_any
                .call_method1("prettily", (error_arg,))?
                .extract()?;
            ret.push(indent(&pretty));
        }
        Ok(ret.join("\n"))
    }

    fn __str__(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<String> {
        Node::prettily(slf, py, None)
    }

    fn __repr__(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<String> {
        fn repr_node(py: Python<'_>, node: &Bound<'_, PyAny>, top_level: bool) -> PyResult<String> {
            let full_text = node.getattr("full_text")?;
            let class_name = node.get_type().name()?.to_string();
            let expr = node.getattr("expr")?;
            let start: usize = node.getattr("start")?.extract()?;
            let end: usize = node.getattr("end")?.extract()?;
            let children = node.getattr("children")?;

            let mut lines: Vec<String> = Vec::new();
            if top_level {
                let full_text_repr = full_text.repr()?.to_str()?.to_string();
                lines.push(format!("s = {full_text_repr}"));
            }

            let expr_repr = expr.repr()?.to_str()?.to_string();
            let children_list: &Bound<'_, PyList> = children.downcast::<PyList>()?;
            if children_list.len() == 0 {
                lines.push(format!("{class_name}({expr_repr}, s, {start}, {end})"));
                return Ok(lines.join("\n"));
            }

            let mut child_reprs: Vec<String> = Vec::with_capacity(children_list.len());
            for c in children_list.iter() {
                child_reprs.push(repr_node(py, &c, false)?);
            }
            let joined = child_reprs.join(", ");
            lines.push(format!(
                "{class_name}({expr_repr}, s, {start}, {end}, children=[{joined}])"
            ));
            Ok(lines.join("\n"))
        }

        let self_any = unsafe { Bound::from_borrowed_ptr(py, slf.as_ptr()) }.into_any();
        repr_node(py, &self_any, true)
    }

    fn __richcmp__(
        &self,
        py: Python<'_>,
        other: Py<PyAny>,
        op: pyo3::basic::CompareOp,
    ) -> PyResult<Py<PyAny>> {
        let other_any = other.bind(py);
        let is_node = other_any.is_instance_of::<Node>();
        if is_node == false {
            return Ok(py.NotImplemented());
        }

        let other_node: PyRef<'_, Node> = other_any.extract()?;

        let expr_eq = self.expr.bind(py).eq(other_node.expr.bind(py))?;
        let text_eq = self.full_text.bind(py).eq(other_node.full_text.bind(py))?;
        let start_eq = self.start == other_node.start;
        let end_eq = self.end == other_node.end;
        let children_eq = self.children.bind(py).eq(other_node.children.bind(py))?;

        let is_equal = expr_eq && text_eq && start_eq && end_eq && children_eq;
        let out = match op {
            pyo3::basic::CompareOp::Eq => is_equal,
            pyo3::basic::CompareOp::Ne => !is_equal,
            _ => {
                return Ok(py.NotImplemented());
            }
        };
        Ok(out.into_pyobject(py)?.to_owned().into_any().unbind())
    }
}

#[pymethods]
impl RegexNode {
    #[new]
    #[pyo3(signature = (expr, full_text, start, end, children=None, match_obj=None))]
    fn new(
        py: Python<'_>,
        expr: Py<PyAny>,
        full_text: Py<PyAny>,
        start: usize,
        end: usize,
        children: Option<Py<PyList>>,
        match_obj: Option<Py<PyAny>>,
    ) -> PyResult<(Self, Node)> {
        let base = Node::new(py, expr, full_text, start, end, children)?;
        let m = match match_obj {
            Some(m) => m,
            None => py.None(),
        };
        Ok((Self { r#match: m }, base))
    }
}
