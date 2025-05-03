use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use nalgebra::DVector;

#[pyfunction]
fn dot_product(a: Vec<f64>, b: Vec<f64>) -> PyResult<f64> {
    if a.len() != b.len() {
        return Err(PyValueError::new_err("Vectors must be the same length"));
    }

    let va = DVector::from_vec(a);
    let vb = DVector::from_vec(b);

    Ok(va.dot(&vb))
}

#[pymodule]
fn dot_product(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dot_product, m)?)?;
    Ok(())
}
