import numpy as np

def simple_iteration_method(A, b, eps):
    n = len(A)
    x = np.zeros(n)
    converge = False
    while not converge:
        x_new = np.dot(np.linalg.inv(A), b)
        converge = np.sqrt(np.dot(x - x_new, x - x_new)) <= eps
        x = x_new
    return x

# Ваша система уравнений
A = np.array([[-26, -7, -8, -2], [2, -17, -6, -2], [-7, -6, -23, -3], [3, -2, -7, -13]], dtype=float)
b = np.array([-51, 85, -71, 91], dtype=float)
eps = 1e-6

x = simple_iteration_method(A, b, eps)

print("Решение системы уравнений: ", x)
##---
def seidel_method(A, b, eps):
    n = len(A)
    x = np.zeros(n)
    converge = False
    while not converge:
        x_new = np.copy(x)
        for i in range(n):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]
        converge = np.sqrt(np.dot(x - x_new, x - x_new)) <= eps
        x = x_new
    return x

# Ваша система уравнений
A = np.array([[-26, -7, -8, -2], [2, -17, -6, -2], [-7, -6, -23, -3], [3, -2, -7, -13]], dtype=float)
b = np.array([-51, 85, -71, 91], dtype=float)
eps = 1e-6

x = seidel_method(A, b, eps)

print("Решение системы уравнений: ", x)
