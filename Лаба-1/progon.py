def tridiagonal_matrix_algorithm(A, d):
    """
    Метод прогонки (алгоритм Томаса) для решения СЛАУ с трёхдиагональной матрицей
    A - трёхдиагональная матрица (список списков)
    d - вектор правой части
    """
    n = len(d)

    # выделяем поддиагональ, диагональ и наддиагональ
    a = [0.0] * n
    b = [0.0] * n
    c = [0.0] * n
    for i in range(n):
        b[i] = A[i][i]
        if i > 0:
            a[i] = A[i][i-1]
        if i < n-1:
            c[i] = A[i][i+1]

    # прогоночные коэффициенты
    p = [0.0] * n
    q = [0.0] * n

    # прямой ход
    p[0] = -c[0] / b[0]
    q[0] = d[0] / b[0]
    for i in range(1, n):
        denom = b[i] + a[i] * p[i-1]
        p[i] = -c[i] / denom if i < n-1 else 0.0
        q[i] = (d[i] - a[i] * q[i-1]) / denom

    # обратный ход
    x = [0.0] * n
    x[-1] = q[-1]
    for i in range(n-2, -1, -1):
        x[i] = p[i] * x[i+1] + q[i]

    return x


def determinant_tridiagonal(A):
    """
    Определитель трёхдиагональной матрицы
    """
    n = len(A)

    # выделяем диагонали
    a = [0.0] * n
    b = [0.0] * n
    c = [0.0] * n
    for i in range(n):
        b[i] = A[i][i]
        if i > 0:
            a[i] = A[i][i-1]
        if i < n-1:
            c[i] = A[i][i+1]

    # рекуррентное вычисление определителя
    if n == 1:
        return b[0]
    if n == 2:
        return b[0] * b[1] - a[1] * c[0]

    det_prev_prev = 1.0
    det_prev = b[0]
    for i in range(2, n+1):
        det_current = b[i-1] * det_prev - a[i-1] * c[i-2] * det_prev_prev
        det_prev_prev, det_prev = det_prev, det_current
    return det_prev

def test_tridiagonal_method():
    A = [
        [7, -3, 0, 0, 0, 0, 0, 0],
        [3,  5, -2, 0, 0, 0, 0, 0],
        [0,  2,  9, -1, 0, 0, 0, 0],
        [0,  0, -2,  7, -3, 0, 0, 0],
        [0,  0,  0,  3,  8, 1, 0, 0],
        [0,  0,  0,  0, -5, 9, 4, 0],
        [0,  0,  0,  0,  0,  3, -6, -2],
        [0,  0,  0,  0,  0,  0,  3, 8]
    ]
    d = [26, 28, 15, 7, -23, 24, -3, 24]

    print("=== Решение методом прогонки ===")
    x = tridiagonal_matrix_algorithm(A, d)
    for i, xi in enumerate(x, 1):
        print(f"x{i} = {xi:.6f}")

    det = determinant_tridiagonal(A)
    print(f"\nОпределитель матрицы: {det:.6f}")

if __name__ == "__main__":
    test_tridiagonal_method()
