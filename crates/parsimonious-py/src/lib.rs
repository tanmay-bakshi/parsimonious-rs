#![allow(clippy::needless_pass_by_value)]

use pyo3::prelude::*;

mod cache;
mod expr;
mod node;

#[pymodule]
#[pyo3(name = "_parsimonious_rs")]
fn _parsimonious_rs(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<cache::PackratCache>()?;

    m.add_class::<expr::Expression>()?;
    m.add_class::<expr::Literal>()?;
    m.add_class::<expr::TokenMatcher>()?;
    m.add_class::<expr::Regex>()?;
    m.add_class::<expr::Sequence>()?;
    m.add_class::<expr::OneOf>()?;
    m.add_class::<expr::Lookahead>()?;
    m.add_class::<expr::Quantifier>()?;

    m.add_class::<node::Node>()?;
    m.add_class::<node::RegexNode>()?;

    Ok(())
}
