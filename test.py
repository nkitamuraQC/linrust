import numpy as np
import pytest
from linrust import *

def test_dot():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    assert dot(a, b) == pytest.approx(np.dot(a, b))

def test_normalize():
    a = np.array([3.0, 4.0])
    result = normalize(a)
    expected = a / np.linalg.norm(a)
    assert np.allclose(result, expected)

def test_matmul():
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[5.0, 6.0], [7.0, 8.0]])
    result = matmul(A, B)
    expected = A @ B
    assert np.allclose(result, expected)

def test_transpose():
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = transpose(A)
    expected = A.T
    assert np.allclose(result, expected)

def test_determinant():
    A = np.array([[4.0, 7.0], [2.0, 6.0]])
    result = determinant(A)
    expected = np.linalg.det(A)
    assert result == pytest.approx(expected)

def test_inverse():
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = inverse(A)
    expected = np.linalg.inv(A)
    assert np.allclose(result, expected)

def test_diagonalize():
    A = np.array([[2.0, 0.0], [0.0, 3.0]])
    vals, vecs = diagonalize(A)
    w, v = np.linalg.eig(A)
    assert np.allclose(sorted(vals), sorted(w))
    # Eigenvectors may differ by sign or order, so skip strict test

def test_qr_decompose():
    A = np.array([[12.0, -51.0], [6.0, 167.0]])
    q, r = qr_decompose(A)
    Q, R = np.linalg.qr(A)
    assert np.allclose(np.abs(q), np.abs(Q), atol=1e-5)  # allow sign flip
    assert np.allclose(r, R, atol=1e-5)

def test_svd_decompose():
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    u, s, vt = svd_decompose(A)
    U, S, VT = np.linalg.svd(A)
    assert np.allclose(np.abs(u), np.abs(U), atol=1e-5)
    assert np.allclose(s, S, atol=1e-5)
    assert np.allclose(np.abs(vt), np.abs(VT), atol=1e-5)

if __name__ == "__main__":
    test_dot()
    print("Dot product test passed!")
    test_normalize()
    print("Normalization test passed!")
    test_matmul()
    print("Matrix multiplication test passed!")
    test_transpose()
    print("Transpose test passed!")
    test_determinant()
    print("Determinant test passed!")
    test_inverse()
    print("Inverse test passed!")
    test_diagonalize()
    print("Diagonalization test passed!")
    test_qr_decompose()
    print("QR decomposition test passed!")
    test_svd_decompose()
    print("All tests passed!")