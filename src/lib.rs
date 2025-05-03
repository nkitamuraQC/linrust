use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

/// Pythonから呼び出せる関数：2つの同じ長さの配列のドット積を計算
#[pyfunction]
fn dot_product(a: Vec<f64>, b: Vec<f64>) -> PyResult<f64> {
    if a.len() != b.len() {
        return Err(PyValueError::new_err("Vectors must be the same length"));
    }
    let result = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    Ok(result)
}

/// Pythonモジュールとしてエクスポート
#[pymodule]
fn dot_product(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dot_product, m)?)?;
    Ok(())
}
