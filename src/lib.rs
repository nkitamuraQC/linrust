use ndarray::{Array1, Array2};
use numpy::Complex64;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use ndarray_linalg::{
    DeterminantInto, Eig, Inverse, Norm, QRInto, SVDInto,
};

#[pyfunction]
fn dot(a: PyReadonlyArray1<f64>, b: PyReadonlyArray1<f64>) -> PyResult<f64> {
    Ok(a.as_array().dot(&b.as_array()))
}

#[pyfunction]
fn normalize<'py>(py: Python<'py>, a: PyReadonlyArray1<f64>) -> PyResult<&'py PyArray1<f64>> {
    let a = a.as_array();
    let norm = a.norm_l2();
    Ok((a.to_owned() / norm).into_pyarray(py))
}

#[pyfunction]
fn matmul<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<f64>,
    b: PyReadonlyArray2<f64>,
) -> PyResult<&'py PyArray2<f64>> {
    let a = a.as_array();
    let b = b.as_array();
    Ok(a.dot(&b).into_pyarray(py))
}

#[pyfunction]
fn transpose<'py>(py: Python<'py>, a: PyReadonlyArray2<f64>) -> PyResult<&'py PyArray2<f64>> {
    Ok(a.as_array().t().to_owned().into_pyarray(py))
}

#[pyfunction]
fn determinant(a: PyReadonlyArray2<f64>) -> PyResult<f64> {
    a.as_array()
        .to_owned()
        .det_into()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("det error: {:?}", e)))
}

#[pyfunction]
fn inverse<'py>(py: Python<'py>, a: PyReadonlyArray2<f64>) -> PyResult<&'py PyArray2<f64>> {
    a.as_array()
        .to_owned()
        .inv()
        .map(|inv| inv.into_pyarray(py))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("inv error: {:?}", e)))
}

#[pyfunction]
fn eigen<'py>(py: Python<'py>, a: PyReadonlyArray2<f64>) -> PyResult<(&'py PyArray1<Complex64>, &'py PyArray2<Complex64>)> {
    let a = a.as_array().to_owned();
    let (values, vectors) = a
        .eig()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("eigen error: {:?}", e)))?;
    Ok((values.into_pyarray(py), vectors.into_pyarray(py)))
}

/*
#[pyfunction]
fn lu_decomposition<'py>(py: Python<'py>, a: PyReadonlyArray2<f64>) -> PyResult<(&'py PyArray2<f64>, &'py PyArray2<f64>)> {
    let (l, u) = a
        .as_array()
        .to_owned()
        .lu_into()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("LU error: {:?}", e)))?;
    Ok((l.into_pyarray(py), u.into_pyarray(py)))
}
*/

#[pyfunction]
fn qr_decomposition<'py>(py: Python<'py>, a: PyReadonlyArray2<f64>) -> PyResult<(&'py PyArray2<f64>, &'py PyArray2<f64>)> {
    let (q, r) = a
        .as_array()
        .to_owned()
        .qr_into()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("QR error: {:?}", e)))?;
    Ok((q.into_pyarray(py), r.into_pyarray(py)))
}

#[pyfunction]
fn svd_decomposition<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<f64>,
) -> PyResult<(&'py PyArray2<f64>, &'py PyArray1<f64>, &'py PyArray2<f64>)> {
    let (u_opt, s, vt_opt) = a
        .as_array()
        .to_owned()
        .svd_into(true, true)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("SVD error: {:?}", e)))?;

    let u = u_opt.ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing U matrix from SVD"))?;
    let vt = vt_opt.ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing VT matrix from SVD"))?;

    Ok((u.into_pyarray(py), s.into_pyarray(py), vt.into_pyarray(py)))
}

#[pymodule]
fn linrust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dot, m)?)?;
    m.add_function(wrap_pyfunction!(normalize, m)?)?;
    m.add_function(wrap_pyfunction!(matmul, m)?)?;
    m.add_function(wrap_pyfunction!(transpose, m)?)?;
    m.add_function(wrap_pyfunction!(determinant, m)?)?;
    m.add_function(wrap_pyfunction!(inverse, m)?)?;
    m.add_function(wrap_pyfunction!(eigen, m)?)?;
    // m.add_function(wrap_pyfunction!(lu_decomposition, m)?)?;
    m.add_function(wrap_pyfunction!(qr_decomposition, m)?)?;
    m.add_function(wrap_pyfunction!(svd_decomposition, m)?)?;
    Ok(())
}