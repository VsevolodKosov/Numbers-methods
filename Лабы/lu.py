import numpy as np

def lu_decomposition(A):

    n = len(A)
    
    L = np.zeros((n,n))
    U = np.zeros((n,n))
    P = np.eye(n)
    
    for i in range(n):
        max_row_index = np.argmax(abs(A[i:, i])) + i
        P[[i, max_row_index]] = P[[max_row_index, i]]
        A[[i, max_row_index]] = A[[max_row_index, i]]
        
        for j in range(i+1, n):
            factor = A[j, i] / A[i, i]
            L[j, i] = factor
            A[j, i:] = A[j, i:] - factor * A[i, i:]
            
        U[i, i:] = A[i, i:]
        
    L += np.eye(n)
    return P, L, U

# Ваша система уравнений
A = np.array([[-2, -1, -9, -5], [-4, 4, -2, 6], [5, 7, -4, 1], [9, 7, 7, 1]], dtype=float)
b = np.array([93, -16, -80, -119], dtype=float)

P, L, U = lu_decomposition(A.copy())

# Решение системы уравнений
y = np.linalg.solve(L, P @ b)
x = np.linalg.solve(U, y)

# Определитель матрицы
det = np.prod(np.diag(U))

# Обратная матрица
inv_A = np.linalg.inv(A)

print("Решение системы уравнений: ", x)
print("Определитель матрицы: ", det)
print("Обратная матрица: ", inv_A)
