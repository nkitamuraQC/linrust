import numpy as np
import ctypes
from ctypes import c_double, c_size_t, POINTER

# 共有ライブラリを読み込む（環境に合わせて変更）
lib = ctypes.CDLL("./target/release/liblinrust.dylib")

# ---- dot product ----
lib.dot_product.argtypes = [POINTER(c_double), POINTER(c_double), c_size_t]
lib.dot_product.restype = c_double

def dot(a: np.ndarray, b: np.ndarray) -> float:
    return lib.dot_product(
        a.ctypes.data_as(POINTER(c_double)),
        b.ctypes.data_as(POINTER(c_double)),
        a.size
    )

# ---- normalize ----
lib.normalize.argtypes = [POINTER(c_double), c_size_t, POINTER(c_double)]
lib.normalize.restype = None

def normalize(a: np.ndarray) -> np.ndarray:
    out = np.zeros_like(a)
    lib.normalize(
        a.ctypes.data_as(POINTER(c_double)),
        a.size,
        out.ctypes.data_as(POINTER(c_double))
    )
    return out

# ---- matmul ----
lib.matmul.argtypes = [
    POINTER(c_double), POINTER(c_double),
    c_size_t, c_size_t, c_size_t,
    POINTER(c_double)
]
lib.matmul.restype = None

def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    assert a.shape[1] == b.shape[0]
    out = np.zeros((a.shape[0], b.shape[1]), dtype=np.float64)
    lib.matmul(
        a.ctypes.data_as(POINTER(c_double)),
        b.ctypes.data_as(POINTER(c_double)),
        a.shape[0], a.shape[1], b.shape[1],
        out.ctypes.data_as(POINTER(c_double))
    )
    return out

# ---- transpose ----
lib.transpose.argtypes = [POINTER(c_double), c_size_t, c_size_t, POINTER(c_double)]
lib.transpose.restype = None

def transpose(a: np.ndarray) -> np.ndarray:
    result = np.zeros_like(a)  # 出力用の行列を初期化
    lib.transpose(a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
              a.shape[0], a.shape[1], result.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

# 結果を表示
    return result

# ---- determinant ----
lib.determinant.argtypes = [POINTER(c_double), c_size_t]
lib.determinant.restype = c_double

def determinant(a: np.ndarray) -> float:
    assert a.shape[0] == a.shape[1]
    return lib.determinant(
        a.ctypes.data_as(POINTER(c_double)),
        a.shape[0]
    )

# ---- inverse ----
lib.inverse.argtypes = [POINTER(c_double), c_size_t, POINTER(c_double)]
lib.inverse.restype = None

def inverse(a: np.ndarray) -> np.ndarray:
    n = a.shape[0]
    out = np.zeros((n, n), dtype=np.float64)
    lib.inverse(
        a.ctypes.data_as(POINTER(c_double)),
        n,
        out.ctypes.data_as(POINTER(c_double))
    )
    return out

# ---- diagonalize ----
lib.diagonalize.argtypes = [POINTER(c_double), c_size_t, POINTER(c_double), POINTER(c_double)]
lib.diagonalize.restype = None

def diagonalize(a: np.ndarray):
    n = a.shape[0]
    eigvals = np.zeros(n, dtype=np.float64)
    eigvecs = np.zeros((n, n), dtype=np.float64)
    lib.diagonalize(
        a.ctypes.data_as(POINTER(c_double)),
        n,
        eigvals.ctypes.data_as(POINTER(c_double)),
        eigvecs.ctypes.data_as(POINTER(c_double))
    )
    return eigvals, eigvecs

# ---- QR decomposition ----
lib.qr_decompose.argtypes = [POINTER(c_double), c_size_t, c_size_t, POINTER(c_double), POINTER(c_double)]
lib.qr_decompose.restype = None

def qr_decompose(a: np.ndarray):
    rows, cols = a.shape
    q = np.zeros((rows, cols), dtype=np.float64)
    r = np.zeros((cols, cols), dtype=np.float64)
    lib.qr_decompose(
        a.ctypes.data_as(POINTER(c_double)),
        rows, cols,
        q.ctypes.data_as(POINTER(c_double)),
        r.ctypes.data_as(POINTER(c_double))
    )
    return q, r

# ---- SVD decomposition ----
lib.svd_decompose.argtypes = [POINTER(c_double), c_size_t, c_size_t, POINTER(c_double), POINTER(c_double), POINTER(c_double)]
lib.svd_decompose.restype = None

def svd_decompose(a: np.ndarray):
    rows, cols = a.shape
    u = np.zeros((rows, rows), dtype=np.float64)
    s = np.zeros(min(rows, cols), dtype=np.float64)
    vt = np.zeros((cols, cols), dtype=np.float64)
    lib.svd_decompose(
        a.ctypes.data_as(POINTER(c_double)),
        rows, cols,
        u.ctypes.data_as(POINTER(c_double)),
        s.ctypes.data_as(POINTER(c_double)),
        vt.ctypes.data_as(POINTER(c_double))
    )
    return u, s, vt