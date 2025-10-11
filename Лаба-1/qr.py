import math

def qr_eigenvalues(A, epsilon=1e-7):
    n = len(A)
    A_k = [row[:] for row in A]
    eigenvalues_prev = [float('inf')] * n
    iterations = 0

    while True:
        iterations += 1
        # --- QR-разложение через отражения Хаусхолдера ---
        Q = [[1.0 if i==j else 0.0 for j in range(n)] for i in range(n)]
        R = [row[:] for row in A_k]

        for k in range(n-1):
            # Вектор v
            v = [0.0]*n
            norm = math.sqrt(sum(R[i][k]**2 for i in range(k, n)))
            sign = 1.0 if R[k][k] >= 0 else -1.0
            v[k] = R[k][k] + sign*norm
            for i in range(k+1, n):
                v[i] = R[i][k]
            v_norm = sum(v[i]**2 for i in range(k, n))
            if v_norm < 1e-15:
                continue
            # H = I - 2 vv^T / ||v||^2
            H = [[1.0 if i==j else 0.0 for j in range(n)] for i in range(n)]
            for i in range(n):
                for j in range(n):
                    H[i][j] -= 2*v[i]*v[j]/v_norm
            # R = H*R
            R_new = [[sum(H[i][k]*R[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
            R = R_new
            # Q = Q*H
            Q_new = [[sum(Q[i][k]*H[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
            Q = Q_new

        # A_k+1 = R*Q
        A_next = [[sum(R[i][k]*Q[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
        A_k = A_next

        # --- Вычисление собственных значений ---
        eigenvalues = []
        i = 0
        while i < n:
            if i < n-1 and abs(A_k[i+1][i]) > epsilon:
                a,b = A_k[i][i], A_k[i][i+1]
                c,d = A_k[i+1][i], A_k[i+1][i+1]
                trace, det = a+d, a*d - b*c
                disc = trace**2 - 4*det
                if disc < 0:
                    real = trace/2
                    imag = math.sqrt(-disc)/2
                    eigenvalues.append(complex(real, imag))
                    eigenvalues.append(complex(real, -imag))
                else:
                    eigenvalues.append((trace+math.sqrt(disc))/2)
                    eigenvalues.append((trace-math.sqrt(disc))/2)
                i += 2
            else:
                eigenvalues.append(A_k[i][i])
                i += 1

        # Проверка сходимости
        if all(abs(eigenvalues[i]-eigenvalues_prev[i])<epsilon for i in range(n)):
            break
        eigenvalues_prev = eigenvalues

    # --- Проверка det(A - λI) ---
    checks = []
    for eig in eigenvalues:
        A_minus_lambda_I = [[A[i][j] - (eig if i==j else 0) for j in range(n)] for i in range(n)]
        temp = [row[:] for row in A_minus_lambda_I]
        det = 1.0
        for col in range(n):
            max_row = max(range(col,n), key=lambda r: abs(temp[r][col]))
            if abs(temp[max_row][col])<1e-10:
                det = 0.0
                break
            if max_row != col:
                temp[col], temp[max_row] = temp[max_row], temp[col]
                det *= -1
            det *= temp[col][col]
            for row in range(col+1, n):
                factor = temp[row][col]/temp[col][col]
                for j in range(col, n):
                    temp[row][j] -= factor*temp[col][j]
        checks.append(abs(det))
    return eigenvalues, iterations, checks

if __name__ == "__main__":
    A = [
        [3, -5, -4, 7, -1],
        [-1, 17, 1, 2, 2],
        [-2, 3, 4, -1, 5],
        [2, -1, -4, 1, 3],
        [1, 3, -5, 1, 2]
    ]
    EPS = 1e-7
    eigenvalues, iterations, det_checks = qr_eigenvalues(A, EPS)

    print("Собственные значения матрицы:")
    for idx, eig in enumerate(eigenvalues):
        print(f"λ_{idx+1} = {eig}")

    print(f"\nКоличество итераций: {iterations}")

    print("\nПроверка корректности собственных значений:")
    for idx, det_val in enumerate(det_checks):
        print(f"|det(A - λ_{idx+1}I)| = {det_val:.2e} {'OK' if det_val<1e-1 else 'WRONG'}")
