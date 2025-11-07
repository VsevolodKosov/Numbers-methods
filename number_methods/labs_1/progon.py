def progon(A, d):
    n = len(d)

    a = [0.0] * n
    b = [0.0] * n
    c = [0.0] * n
    for i in range(n):
        b[i] = A[i][i]
        if i > 0:
            a[i] = A[i][i-1]
        if i < n-1:
            c[i] = A[i][i+1]

    stability = False
    for i in range(n):
        left_side = abs(b[i])
        right_side = abs(a[i]) + abs(c[i])
        if left_side > right_side:
            stability = True
            break

    p = [0.0] * n
    q = [0.0] * n

    p[0] = -c[0] / b[0]
    q[0] = d[0] / b[0]
    
    check = True
    if abs(p[0]) >= 1:
        check = False

    for i in range(1, n):
        denom = b[i] + a[i] * p[i-1]
        p[i] = -c[i] / denom if i < n-1 else 0.0
        q[i] = (d[i] - a[i] * q[i-1]) / denom
        
        if i < n-1 and abs(p[i]) >= 1:
            check = False

    x = [0.0] * n
    x[-1] = q[-1]
    for i in range(n-2, -1, -1):
        x[i] = p[i] * x[i+1] + q[i]

    return x, stability, check


def determinant_tridiagonal(A):
    n = len(A)
  
    a = [0.0] * n
    b = [0.0] * n
    c = [0.0] * n
    for i in range(n):
        b[i] = A[i][i]
        if i > 0:
            a[i] = A[i][i-1]
        if i < n-1:
            c[i] = A[i][i+1]

    p = [0.0] * n
    if n > 1:
        p[0] = -c[0] / b[0]
    
    for i in range(1, n):
        denom = b[i] + a[i] * p[i-1]
        p[i] = -c[i] / denom if i < n-1 else 0.0

    determinant = 1.0
    for i in range(n):
        if i == 0:
            determinant *= b[0]
        else:
            determinant *= b[i] + a[i] * p[i-1]

    return determinant


if __name__ == "__main__":
    A = [
        [7, -3, 0, 0, 0, 0, 0, 0],
        [3, 5, -2, 0, 0, 0, 0, 0],
        [0, 2, 9, -1, 0, 0, 0, 0],
        [0, 0, -2, 7, -3, 0, 0, 0],
        [0, 0, 0, 3, 8, 1, 0, 0],
        [0, 0, 0, 0, -5, 9, 4, 0],
        [0, 0, 0, 0, 0, 3, -6, -2],
        [0, 0, 0, 0, 0, 0, 3, 8]
    ]
    
    d = [26, 28, 15, 7, -23, 24, -3, 24]

    x, stability, check = progon(A, d)
    
    print("=== Решение методом прогонки ===")
    for i in range(len(x)):
        print(f"x{i+1} = {x[i]:.6f}")
    
    print(f"Устойчивость - {str(stability).lower()}")
    print(f"Проверка - {str(check).lower()}")
    
    det = determinant_tridiagonal(A)
    print(f"\nОпределитель матрицы: {det:.6f}")