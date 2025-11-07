def method_Gaus(A, b):
    n = len(A)

    augmented = []
    for i in range(n):
        row = A[i][:] + [1 if j == i else 0 for j in range(n)] + [b[i]]
        augmented.append(row)
    cols = 2 * n + 1

    det_factor = 1.0
    for col in range(n):
        pivot = augmented[col][col] #a_11

        if abs(pivot) < 1e-12:
            det = 0.0
            inv = None
            sol = None
            return det, inv, sol

        det_factor *= pivot
        for j in range(col, cols):
            augmented[col][j] /= pivot #a_1j / a_11

        for i in range(n):
            if i == col:
                continue
            factor = augmented[i][col] # a_i1
            for j in range(col, cols):
                augmented[i][j] -= factor * augmented[col][j] #a_i1 * a_1j

    det = det_factor

    inv = []
    for i in range(n):
        inv.append(augmented[i][n:2 * n])
    sol = []
    for i in range(n):
        sol.append(augmented[i][2 * n])

    return det, inv, sol


if __name__ == "__main__":
    A = [
        [3, 6, -4, 3, 2],
        [4, 2, 1, 3, 5],
        [-2, 3, 3, 2, 9],
        [2, -5, -4, 0, 3],
        [9, -4, 5, 1, -2]
    ]
    b = [15, 58, 72, 39, 24]

    det, inv, sol = method_Gaus(A, b)

    print("Результаты метода Гаусса:")
    print("\nРешение системы:")
    print([x for x in sol])

    print(f"\nОпределитель: {det}")

    print("\nОбратная матрица:")
    for row in inv:
        print([x for x in row])
