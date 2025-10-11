import numpy as np

A = [
    [3, 6, -4, 3, 2],
    [4, 2, 1, 3, 5],
    [-2, 3, 3, 2, 9],
    [2, -5, -4, 0, 3],
    [9, -4, 5, 1, -2]
]
b = [15, 58, 72, 39, 24]

def gaussian_matrix_operations(A, b):
    n = len(A)

    # Создаем расширенную матрицу [A | I | b] (если b задан)
    augmented = []
    for i in range(n):
        row = A[i][:] + [1 if j == i else 0 for j in range(n)] + [b[i]]
        augmented.append(row)
    cols = 2 * n + 1

    for i in augmented:
        print(i)
    print()

    swaps = 0
    det_factor = 1.0

    for col in range(n):
        max_row = col
        for i in range(col + 1, n):
            if abs(augmented[i][col]) > abs(augmented[max_row][col]):
                max_row = i

        if max_row != col:
            augmented[col], augmented[max_row] = augmented[max_row], augmented[col]
            swaps += 1
            det_factor *= -1

        pivot = augmented[col][col]

        if abs(pivot) < 1e-12:
            det = 0.0
            inv = None
            sol = None
            return det, inv, sol

        det_factor *= pivot

        for j in range(col, cols):
            augmented[col][j] /= pivot

        for i in range(n):
            if i == col:
                continue
            factor = augmented[i][col]
            for j in range(col, cols):
                augmented[i][j] -= factor * augmented[col][j]

    det = det_factor

    inv = []
    for i in range(n):
        inv.append(augmented[i][n:2 * n])

    sol = None
    if b is not None:
        sol = []
        for i in range(n):
            sol.append(augmented[i][2 * n])

    return det, inv, sol


print("Результаты метода Гаусса:")
det, inv, sol = gaussian_matrix_operations(A, b)

print("Решение системы:")
print([round(x, 4) for x in sol])

print(f"\nОпределитель: {round(det, 4)}")

print("\nОбратная матрица:")
for row in inv:
    print([round(x, 4) for x in row])

print("\nПроверка с NumPy:")
A_np = np.array(A)
b_np = np.array(b)

np_solution = np.linalg.solve(A_np, b_np)
print("\nРешение системы:")
print(np_solution.round(4))

np_det = np.linalg.det(A_np)
print(f"\nОпределитель : {round(np_det, 4)}")

np_inv = np.linalg.inv(A_np)
print("\nОбратная матрица:")
print(np_inv.round(4))
