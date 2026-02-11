use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::PyAnyMethods;

pub(crate) enum CacheSlot {
    Unknown,
    InProgress,
    Done(Py<PyAny>), // Node or None
}

#[pyclass]
pub(crate) struct PackratCache {
    pub(crate) len: usize,
    pub(crate) per_expr: HashMap<usize, Vec<CacheSlot>>,
}

#[pymethods]
impl PackratCache {
    #[new]
    fn new(py: Python<'_>, text: Py<PyAny>) -> PyResult<Self> {
        Self::from_text(py, text)
    }
}

impl PackratCache {
    pub(crate) fn from_text(py: Python<'_>, text: Py<PyAny>) -> PyResult<Self> {
        let text_any = text.bind(py);
        let len = match text_any.len() {
            Ok(l) => l,
            Err(_err) => {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "Text must be str, bytes, or a sequence for TokenGrammar.",
                ));
            }
        };

        Ok(Self {
            len,
            per_expr: HashMap::new(),
        })
    }

    pub(crate) fn slots_for_expr_mut(&mut self, expr_id: usize) -> &mut Vec<CacheSlot> {
        self.per_expr.entry(expr_id).or_insert_with(|| {
            let mut slots: Vec<CacheSlot> = Vec::with_capacity(self.len + 1);
            slots.resize_with(self.len + 1, || CacheSlot::Unknown);
            slots
        })
    }
}
