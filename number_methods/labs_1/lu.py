def lu_solve(A, b):
    n = len(A)
    LU = [row[:] for row in A]
    b_copy = b[:]
    
    # LU-разложение
    for k in range(n-1):
        # u_kk^(k) - уже находится в LU[k][k]
        
        # l_rk^(k) = u_rk^(k-1) / u_kk^(k), 
        for r in range(k+1, n):
            LU[r][k] = LU[r][k] / LU[k][k]
        
        # u_rc^(k) = u_rc^(k-1) - l_rk^(k) * u_kc^(k)
        for r in range(k+1, n):
            for c in range(k+1, n):
                LU[r][c] = LU[r][c] - LU[r][k] * LU[k][c]
    
    # Решение системы Ly = b (прямая подстановка)
    y = [0] * n
    for i in range(n):
        s = 0
        for j in range(i):
            s += LU[i][j] * y[j]
        y[i] = b_copy[i] - s
    
    # Решение системы Ux = y (обратная подстановка)
    x = [0] * n
    for i in range(n-1, -1, -1):
        s = 0
        for j in range(i+1, n):
            s += LU[i][j] * x[j]
        x[i] = (y[i] - s) / LU[i][i]
    
    # Вычисление определителя
    determinant = 1
    for i in range(n):
        determinant *= LU[i][i]
    
    # Вычисление обратной матрицы
    A_inv = [[0 for _ in range(n)] for _ in range(n)]
    for col in range(n):
        # Решаем систему для каждого столбца единичной матрицы
        e = [1 if i == col else 0 for i in range(n)]
        
        # Прямая подстановка для L
        y_col = [0] * n
        for i in range(n):
            s = 0
            for j in range(i):
                s += LU[i][j] * y_col[j]
            y_col[i] = e[i] - s
        
        # Обратная подстановка для U
        x_col = [0] * n
        for i in range(n-1, -1, -1):
            s = 0
            for j in range(i+1, n):
                s += LU[i][j] * x_col[j]
            x_col[i] = (y_col[i] - s) / LU[i][i]
        
        # Записываем результат как столбец обратной матрицы
        for i in range(n):
            A_inv[i][col] = x_col[i]
    
    return x, LU, b_copy, determinant, A_inv


if __name__ == "__main__":
    mas = [
        [3, 6, -4, 3, 2],
        [4, 2, 1, 3, 5],
        [-2, 3, 3, 2, 9],
        [2, -5, -4, 0, 3],
        [9, -4, 5, 1, -2]
    ]
    b = [15, 58, 72, 39, 24]

    x, LU, b_copy, det, A_inv = lu_solve(mas, b)

    def print_matrix(mat, name):
        print(f"\n{name}:")
        for row in mat:
            print(row)

    print("\nКомбинированная матрица LU:")
    for row in LU:
        print(row)
    
    print("\nРешение системы x:")
    print(x)
    
    print("\nОпределитель матрицы A:")
    print(det)
    
    print_matrix(A_inv, "Обратная матрица A^-1")