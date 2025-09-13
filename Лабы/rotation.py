import numpy as np

def rotation_method(A, eps):
    n = len(A)
    V = np.eye(n)
    while True:
        i, j = np.unravel_index(np.argmax(np.tril(np.abs(A), -1)), A.shape)
        if np.abs(A[i, j]) < eps:
            break
        phi = 0.5 * np.arctan(2 * A[i, j] / (A[i, i] - A[j, j]))
        U = np.eye(n)
        U[i, i] = np.cos(phi)
        U[j, j] = np.cos(phi)
        U[i, j] = -np.sin(phi)
        U[j, i] = np.sin(phi)
        A = np.dot(U.T, np.dot(A, U))
        V = np.dot(V, U)
    return np.diag(A), V

# Ваша матрица
A = np.array([[-8, 5, -7], [5, 1, 4], [-7, 4, 4]], dtype=float)
eps = 1e-6

eigenvalues, eigenvectors = rotation_method(A, eps)

print("Собственные значения матрицы: ", eigenvalues)
print("Собственные векторы матрицы: ", eigenvectors)
