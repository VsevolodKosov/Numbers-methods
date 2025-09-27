def matrix_mult(A, B):
    """Умножение матриц"""
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    if cols_A != rows_B:
        raise ValueError("Несовместимые размеры матриц для умножения")
    result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]
    return result

def vector_dot(u, v):
    """Скалярное произведение векторов"""
    if len(u) != len(v):
        raise ValueError("Векторы должны иметь одинаковую длину")
    return sum(u[i] * v[i] for i in range(len(u)))

def vector_norm(v):
    """Евклидова норма"""
    return sum(x*x for x in v) ** 0.5

def identity_matrix(n):
    """Единичная матрица n x n"""
    I = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        I[i][i] = 1
    return I

def outer_product(u, v):
    """Внешнее произведение векторов"""
    rows, cols = len(u), len(v)
    result = [[0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            result[i][j] = u[i] * v[j]
    return result

def scalar_mult(c, A):
    """Умножение матрицы на скаляр"""
    rows, cols = len(A), len(A[0])
    result = [[0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            result[i][j] = c * A[i][j]
    return result

def matrix_sub(A, B):
    """Вычитание матриц"""
    rows, cols = len(A), len(A[0])
    if rows != len(B) or cols != len(B[0]):
        raise ValueError("Матрицы должны иметь одинаковый размер")
    result = [[0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            result[i][j] = A[i][j] - B[i][j]
    return result

def householder_qr(A):
    """QR-разложение с помощью отражений Хаусхолдера"""
    n = len(A)
    Q = identity_matrix(n)
    R = [row[:] for row in A]

    for k in range(n - 1):
        x = [R[i][k] for i in range(k, n)]
        norm_x = vector_norm(x)
        if norm_x < 1e-15:
            continue
        sign = 1 if x[0] >= 0 else -1
        v = x[:]
        v[0] += sign * norm_x
        beta = 2.0 / vector_dot(v, v)
        v_vt = outer_product(v, v)
        H = matrix_sub(identity_matrix(len(v)), scalar_mult(beta, v_vt))

        # Применяем H к подматрице R
        R_sub = [[R[i][j] for j in range(k, n)] for i in range(k, n)]
        H_R = matrix_mult(H, R_sub)
        for i in range(k, n):
            for j in range(k, n):
                R[i][j] = H_R[i - k][j - k]

        # Обновляем Q
        Q_sub = [[Q[i][j] for j in range(k, n)] for i in range(n)]
        Q_H = matrix_mult(Q_sub, H)
        for i in range(n):
            for j in range(k, n):
                Q[i][j] = Q_H[i][j - k]

    return Q, R

def qr_algorithm(A, eps=1e-10, max_iter=1000):
    """QR-алгоритм для нахождения формы Шура"""
    n = len(A)
    A_k = [row[:] for row in A]

    for iteration in range(max_iter):
        Q, R = householder_qr(A_k)
        A_next = matrix_mult(R, Q)
        A_k = A_next
        # проверка сходимости по поддиагонали
        converged = True
        for m in range(n - 1):
            sub_norm = vector_norm([A_k[i][m] for i in range(m + 1, n)])
            if sub_norm >= eps:
                converged = False
                break
        if converged:
            break
    return A_k

def extract_eigenvalues_from_schur(schur_form, eps=1e-10):
    """Извлечение собственных значений из верхнетреугольной матрицы Шура"""
    n = len(schur_form)
    eigenvalues = []
    i = 0
    while i < n:
        if i == n - 1 or abs(schur_form[i+1][i]) < eps:
            eigenvalues.append(schur_form[i][i])
            i += 1
        else:
            a = schur_form[i][i]
            b = schur_form[i][i+1]
            c = schur_form[i+1][i]
            d = schur_form[i+1][i+1]
            trace = a + d
            det = a*d - b*c
            discriminant = trace**2 - 4*det
            if discriminant >= 0:
                lambda1 = (trace + discriminant**0.5)/2
                lambda2 = (trace - discriminant**0.5)/2
                eigenvalues.extend([lambda1, lambda2])
            else:
                real_part = trace / 2
                imag_part = (-discriminant)**0.5 / 2
                eigenvalues.extend([complex(real_part, imag_part),
                                    complex(real_part, -imag_part)])
            i += 2
    return eigenvalues

# Пример использования
A = [
    [3, -5, -4, 7, -1],
    [-1, 17, 1, 2, 2],
    [-2, 3, 4, -1, 5],
    [2, -1, -4, 1, 3],
    [1, 3, -5, 1, 2]
]

print("Исходная матрица:")
for row in A:
    print(row)

schur_form = qr_algorithm(A, eps=1e-4, max_iter=10000)

print("\nФорма Шура:")
for row in schur_form:
    print([f"{x:.6f}" for x in row])

our_eigenvalues = extract_eigenvalues_from_schur(schur_form)

print("\nСобственные значения:")
for val in our_eigenvalues:
    if isinstance(val, complex):
        print(f"{val.real:.6f} + {val.imag:.6f}i")
    else:
        print(f"{val:.6f}")
