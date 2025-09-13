import numpy as np

def qr_decomposition(A):
    Q, R = np.linalg.qr(A)
    return Q, R

def qr_algorithm(A, eps):
    n = len(A)
    V = np.eye(n)
    while not np.allclose(np.tril(A, -1), 0, atol=eps):
        Q, R = qr_decomposition(A)
        A = np.dot(R, Q)
        V = np.dot(V, Q)
    return np.diag(A), V

# Ваша матрица
A = np.array([[1], [8]], dtype=float)
eps = 1e-6

eigenvalues, eigenvectors = qr_algorithm(A, eps)

print("Собственные значения матрицы: ", eigenvalues)
print("Собственные векторы матрицы: ", eigenvectors)
