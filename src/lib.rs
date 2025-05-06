use std::ffi::c_double;
use libc::size_t;
use ndarray::{ArrayView1, ArrayView2, Array2};
use ndarray_linalg::{Determinant, Inverse, Eig, QR, SVDDC, Norm};
use ndarray_linalg::svddc::{JobSvd};

#[no_mangle]
pub extern "C" fn dot_product(
    a: *const c_double,
    b: *const c_double,
    len: size_t,
) -> c_double {
    let a_slice = unsafe { std::slice::from_raw_parts(a, len) };
    let b_slice = unsafe { std::slice::from_raw_parts(b, len) };
    ArrayView1::from(a_slice).dot(&ArrayView1::from(b_slice))
}

#[no_mangle]
pub extern "C" fn normalize(
    a: *const c_double,
    len: size_t,
    out: *mut c_double,
) {
    let a_slice = unsafe { std::slice::from_raw_parts(a, len) };
    let mut out_slice = unsafe { std::slice::from_raw_parts_mut(out, len) };

    let vec = ArrayView1::from(a_slice);
    let norm = vec.norm_l2();
    
    if norm == 0.0 {
        // normが0の場合、エラー処理
        return; // normがゼロの場合、正規化できません
    }
    
    for (i, val) in vec.iter().enumerate() {
        out_slice[i] = *val / norm;
    }
}

#[no_mangle]
pub extern "C" fn matmul(
    a: *const c_double,
    b: *const c_double,
    a_rows: size_t,
    a_cols: size_t,
    b_cols: size_t,
    out: *mut c_double,
) {
    let a_slice = unsafe { std::slice::from_raw_parts(a, a_rows * a_cols) };
    let b_slice = unsafe { std::slice::from_raw_parts(b, a_cols * b_cols) };
    let mut out_slice = unsafe { std::slice::from_raw_parts_mut(out, a_rows * b_cols) };

    let a_mat = match ArrayView2::from_shape((a_rows, a_cols), a_slice) {
        Ok(mat) => mat,
        Err(_) => return, // エラー処理
    };

    let b_mat = match ArrayView2::from_shape((a_cols, b_cols), b_slice) {
        Ok(mat) => mat,
        Err(_) => return, // エラー処理
    };

    let result = a_mat.dot(&b_mat);

    // as_slice() が Some でない場合の処理
    match result.as_slice() {
        Some(slice) => out_slice.copy_from_slice(slice),
        None => return, // エラー処理
    }
}

#[no_mangle]
pub extern "C" fn transpose(
    a: *const c_double,
    rows: size_t,
    cols: size_t,
    out: *mut c_double,
) {
    let a_slice = unsafe { std::slice::from_raw_parts(a, rows * cols) };
    let out_slice = unsafe { std::slice::from_raw_parts_mut(out, rows * cols) };

    // 行列を作成
    let a_mat = ArrayView2::from_shape((rows, cols), a_slice).unwrap();
    let t_mat = a_mat.t();  // 転置ビュー（コピーではない）

    // 転置を out_slice に書き出す（要コピー）
    for i in 0..cols {
        for j in 0..rows {
            out_slice[i * rows + j] = t_mat[(i, j)];
        }
    }
}

#[no_mangle]
pub extern "C" fn determinant(
    a: *const c_double,
    n: size_t,
) -> c_double {
    let a_slice = unsafe { std::slice::from_raw_parts(a, n * n) };
    let a_mat = match Array2::from_shape_vec((n, n), a_slice.to_vec()) {
        Ok(mat) => mat,
        Err(_) => return 0.0, // エラー処理
    };

    match a_mat.det() {
        Ok(det) => det,
        Err(_) => 0.0, // エラー処理
    }
}

#[no_mangle]
pub extern "C" fn inverse(
    a: *const c_double,
    n: size_t,
    out: *mut c_double,
) {
    let a_slice = unsafe { std::slice::from_raw_parts(a, n * n) };
    let mut out_slice = unsafe { std::slice::from_raw_parts_mut(out, n * n) };

    let a_mat = match Array2::from_shape_vec((n, n), a_slice.to_vec()) {
        Ok(mat) => mat,
        Err(_) => return, // エラー処理
    };

    match a_mat.inv() {
        Ok(inv) => match inv.as_slice() {
            Some(slice) => out_slice.copy_from_slice(slice),
            None => return, // エラー処理
        },
        Err(_) => return, // エラー処理
    }
}

#[no_mangle]
pub extern "C" fn diagonalize(a: *const f64, n: usize, eigvals: *mut f64, eigvecs: *mut f64) {
    let a_slice = unsafe { std::slice::from_raw_parts(a, n * n) };
    let a_mat = match Array2::from_shape_vec((n, n), a_slice.to_vec()) {
        Ok(mat) => mat,
        Err(_) => return, // エラー処理
    };

    match a_mat.eig() {
        Ok((vals, vecs)) => {
            let eigvals_slice = unsafe { std::slice::from_raw_parts_mut(eigvals, n) };
            let eigvecs_slice = unsafe { std::slice::from_raw_parts_mut(eigvecs, n * n) };

            for i in 0..n {
                eigvals_slice[i] = vals[i].re;
            }

            let flat_vecs = match vecs.as_slice() {
                Some(slice) => slice,
                None => return, // エラー処理
            };

            for i in 0..n * n {
                eigvecs_slice[i] = flat_vecs[i].re;
            }
        }
        Err(_) => return, // エラー処理
    }
}

#[no_mangle]
pub extern "C" fn qr_decompose(
    a: *const c_double,
    rows: size_t,
    cols: size_t,
    q_out: *mut c_double,
    r_out: *mut c_double,
) {
    let a_slice = unsafe { std::slice::from_raw_parts(a, rows * cols) };
    let a_mat = match Array2::from_shape_vec((rows, cols), a_slice.to_vec()) {
        Ok(mat) => mat,
        Err(_) => return, // エラー処理
    };

    match a_mat.qr() {
        Ok((q, r)) => {
            let q_slice = unsafe { std::slice::from_raw_parts_mut(q_out, rows * cols) };
            let r_slice = unsafe { std::slice::from_raw_parts_mut(r_out, cols * cols) };

            match q.as_slice() {
                Some(slice) => q_slice.copy_from_slice(slice),
                None => return, // エラー処理
            }

            match r.as_slice() {
                Some(slice) => r_slice.copy_from_slice(slice),
                None => return, // エラー処理
            }
        }
        Err(_) => return, // エラー処理
    }
}

#[no_mangle]
pub extern "C" fn svd_decompose(
    a: *const f64,
    rows: usize,
    cols: usize,
    u_out: *mut f64,
    s_out: *mut f64,
    vt_out: *mut f64
) {
    let a_slice = unsafe { std::slice::from_raw_parts(a, rows * cols) };
    let a_mat = match Array2::from_shape_vec((rows, cols), a_slice.to_vec()) {
        Ok(mat) => mat,
        Err(_) => return, // エラー処理
    };

    match a_mat.svddc(JobSvd::All) {
        Ok((u_opt, s, vt_opt)) => {
            let u = u_opt.unwrap_or_else(|| Array2::zeros((rows, rows)));
            let vt = vt_opt.unwrap_or_else(|| Array2::zeros((cols, cols)));

            let u_slice = unsafe { std::slice::from_raw_parts_mut(u_out, rows * rows) };
            let s_slice = unsafe { std::slice::from_raw_parts_mut(s_out, std::cmp::min(rows, cols)) };
            let vt_slice = unsafe { std::slice::from_raw_parts_mut(vt_out, cols * cols) };

            match u.as_slice() {
                Some(slice) => u_slice.copy_from_slice(slice),
                None => return, // エラー処理
            }

            match s.as_slice() {
                Some(slice) => s_slice.copy_from_slice(slice),
                None => return, // エラー処理
            }

            match vt.as_slice() {
                Some(slice) => vt_slice.copy_from_slice(slice),
                None => return, // エラー処理
            }
        }
        Err(_) => return, // エラー処理
    }
}