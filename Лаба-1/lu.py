from fractions import Fraction

def lu_solve_and_residual(A, b):
    n = len(A)
    LU = [[Fraction(x) for x in row] for row in A]
    b_frac = [Fraction(x) for x in b]
    P = list(range(n))

    # LU-разложение 
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
            LU[i][j] -= s

        for j in range(i + 1, n):
            s = sum(LU[j][k] * LU[k][i] for k in range(i))
            LU[j][i] = (LU[j][i] - s) / LU[i][i]

    # Разделение на L и U
    L = [[Fraction(0) for _ in range(n)] for _ in range(n)]
    U = [[Fraction(0) for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i > j:
                L[i][j] = LU[i][j]
            else:
                U[i][j] = LU[i][j]
        L[i][i] = Fraction(1)

    # Решение системы
    b_permuted = [b_frac[P[i]] for i in range(n)]
    y = [Fraction(0) for _ in range(n)]
    for i in range(n):
        s = sum(LU[i][j] * y[j] for j in range(i))
        y[i] = b_permuted[i] - s

    x = [Fraction(0) for _ in range(n)]
    for i in range(n - 1, -1, -1):
        s = sum(LU[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (y[i] - s) / LU[i][i]

    # Вычисление невязки
    residual = [Fraction(0) for _ in range(n)]
    for i in range(n):
        for j in range(n):
            residual[i] += A[i][j] * x[j]
        residual[i] -= b_frac[i]
    
    return x, L, U, residual

if __name__ == "__main__":
    mas = [
        [3, 6, -4, 3, 2],
        [4, 2, 1, 3, 5],
        [-2, 3, 3, 2, 9],
        [2, -5, -4, 0, 3],
        [9, -4, 5, 1, -2]
    ]
    b = [15, 58, 72, 39, 24]

    x, L, U, residual = lu_solve_and_residual(mas, b)

    # Вывод результатов
    def print_matrix(mat, name):
        print(f"\n{name}:")
        for row in mat:
            print([round(float(e), 4) for e in row])

    print_matrix(U, "Матрица U")
    print_matrix(L, "Матрица L")
    print("\nРешение системы x:")
    print([round(float(e), 4) for e in x])
    print("\nНевязка (Ax-b):")
    print([round(float(e), 4) for e in residual])
