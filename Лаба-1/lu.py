from fractions import Fraction

def lu_decomposition_with_pivoting(A):
    n = len(A)
    LU = [[Fraction(x) for x in row] for row in A]
    P = list(range(n))

    for i in range(n):
        max_row = i
        for k in range(i, n):
            if abs(LU[k][i]) > abs(LU[max_row][i]):
                max_row = k

        if max_row != i:
            LU[i], LU[max_row] = LU[max_row], LU[i]
            P[i], P[max_row] = P[max_row], P[i]

        for j in range(i, n):
            s = sum(LU[i][k] * LU[k][j] for k in range(i))
            LU[i][j] = LU[i][j] - s

        for j in range(i + 1, n):
            s = sum(LU[j][k] * LU[k][i] for k in range(i))
            LU[j][i] = (LU[j][i] - s) / LU[i][i]

    return P, LU

def solve_lu(P, LU, b):
    n = len(b)
    b_permuted = [b[P[i]] for i in range(n)]

    y = [Fraction(0) for _ in range(n)]
    for i in range(n):
        s = sum(LU[i][j] * y[j] for j in range(i))
        y[i] = b_permuted[i] - s

    x = [Fraction(0) for _ in range(n)]
    for i in range(n - 1, -1, -1):
        s = sum(LU[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (y[i] - s) / LU[i][i]

    return x

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

P, LU = lu_decomposition_with_pivoting(mas)

n = len(LU)
L = [[Fraction(0) for _ in range(n)] for _ in range(n)]
U = [[Fraction(0) for _ in range(n)] for _ in range(n)]

for i in range(n):
    for j in range(n):
        if i > j:
            L[i][j] = LU[i][j]
        else:
            U[i][j] = LU[i][j]
    L[i][i] = Fraction(1)

print("Матрица U")
for row in U:
    print([round(float(x), 4) for x in row])

print("\nМатрица L")
for row in L:
    print([round(float(x), 4) for x in row])

x = solve_lu(P, LU, b_frac)

print("\nРешение системы")
print([round(float(x_i), 4) for x_i in x])

residual = [Fraction(0) for _ in range(len(b))]
for i in range(len(mas)):
    for j in range(len(mas[0])):
        residual[i] += A_frac[i][j] * x[j]
    residual[i] -= b_frac[i]
print("\nНеувязка")
print(residual)