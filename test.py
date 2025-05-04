import numpy as np
import linrust as la
import pytest

A = np.array([[4.0, 2.0], [3.0, 5.0]])
v = np.array([1.0, 2.0])

def test1():
    # Test the linear algebra functions

    assert np.allclose(la.dot(v, v), np.dot(v, v))

def test2():
    # normalize
    assert np.allclose(la.normalize(v), v / np.linalg.norm(v))
    
def test3():
    # matmul
    assert np.allclose(la.matmul(A, A), A @ A)
    
def test4():
    # transpose
    assert np.allclose(la.transpose(A), A.T)
    
def test5():
    # determinant
    assert np.allclose(la.determinant(A), np.linalg.det(A))
    
def test6():
    # inverse
    assert np.allclose(la.inverse(A), np.linalg.inv(A))
    
def test7():
    # eigenvalues
    e1, c1 = la.eigen(A)
    e2, c2 = np.linalg.eig(A)
    # print(e1, e2)
    # eigen値は順序が異なる可能性があるため、集合的に比較
    assert np.allclose(np.sort_complex(e1), np.sort_complex(e2))
    
def _test8():
    # LU decomposition
    L, U = la.lu_decomposition(A)
    from scipy.linalg import lu
    P, L_np, U_np = lu(A)
    assert np.allclose(L, L_np)
    assert np.allclose(U, U_np)
    
def test9():
    # QR decomposition
    Q, R = la.qr_decomposition(A)
    Q_np, R_np = np.linalg.qr(A)
    # 符号の差異に注意（浮動小数点分解の特性）
    assert np.allclose(np.abs(Q), np.abs(Q_np), atol=1e-6)
    assert np.allclose(R, R_np, atol=1e-6)
    
def test10():
    # SVD decomposition
    U_r, S_r, VT_r = la.svd_decomposition(A)
    U_np, S_np, VT_np = np.linalg.svd(A)
    # 並び替えられていない可能性があるが通常一致
    assert np.allclose(np.abs(U_r), np.abs(U_np), atol=1e-6)
    assert np.allclose(S_r, S_np, atol=1e-6)
    assert np.allclose(np.abs(VT_r), np.abs(VT_np), atol=1e-6)
    
    #print("✅ All tests passed.")