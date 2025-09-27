from fractions import Fraction

def lu_decomposition_with_pivoting(A):
    n = len(A)
    A_frac = [[Fraction(x) for x in row] for row in A]
    L = [[Fraction(0) for _ in range(n)] for _ in range(n)]
    U = [[Fraction(0) for _ in range(n)] for _ in range(n)]
    P = list(range(n))

    for i in range(n):
        # выбор главного элемента
        max_row = i
        for k in range(i, n):
            if abs(A_frac[k][i]) > abs(A_frac[max_row][i]):
                max_row = k
        if max_row != i:
            A_frac[i], A_frac[max_row] = A_frac[max_row], A_frac[i]
            P[i], P[max_row] = P[max_row], P[i]

        # вычисление U
        for j in range(i, n):
            s = sum(L[i][k] * U[k][j] for k in range(i))
            U[i][j] = A_frac[i][j] - s

        # диагональ L = 1
        L[i][i] = Fraction(1)

        # вычисление L
        for j in range(i + 1, n):
            s = sum(L[j][k] * U[k][i] for k in range(i))
            L[j][i] = (A_frac[j][i] - s) / U[i][i]

    return P, L, U


def solve_lu(P, L, U, b):
    n = len(b)
    b_permuted = [b[P[i]] for i in range(n)]

    # прямая подстановка (Ly = Pb)
    y = [Fraction(0) for _ in range(n)]
    for i in range(n):
        s = sum(L[i][j] * y[j] for j in range(i))
        y[i] = b_permuted[i] - s

    # обратная подстановка (Ux = y)
    x = [Fraction(0) for _ in range(n)]
    for i in range(n - 1, -1, -1):
        s = sum(U[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (y[i] - s) / U[i][i]

    return x


# Пример
mas = [
    [3, 6, -4, 3, 2],
    [4, 2, 1, 3, 5],
    [-2, 3, 3, 2, 9],
    [2, -5, -4, 0, 3],
    [9, -4, 5, 1, -2]
]

b = [15, 58, 72, 39, 24]

A_frac = [[Fraction(x) for x in row] for row in mas]
b_frac = [Fraction(x) for x in b]

P, L, U = lu_decomposition_with_pivoting(mas)

print("Матрица U:")
for row in U:
    print([round(float(x), 4) for x in row])

print("\nМатрица L:")
for row in L:
    print([round(float(x), 4) for x in row])

x = solve_lu(P, L, U, b_frac)
x_float = [float(x_i) for x_i in x]

print("\nРешение системы:")
print([round(val, 6) for val in x_float])
