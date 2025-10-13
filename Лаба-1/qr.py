import math

def qr_eigenvalues(A, epsilon=1e-7):
    n = len(A)
    A_k = [row[:] for row in A]
    eigenvalues_prev = [float('inf')] * n
    iterations = 0

    while True:
        iterations += 1
        Q = [[1.0 if i==j else 0.0 for j in range(n)] for i in range(n)]
        R = [row[:] for row in A_k]

        for k in range(n-1):
            v = [0.0]*n
            norm = math.sqrt(sum(R[i][k]**2 for i in range(k, n)))
            sign = 1.0 if R[k][k] >= 0 else -1.0
            v[k] = R[k][k] + sign*norm
            for i in range(k+1, n):
                v[i] = R[i][k]
            v_norm = sum(v[i]**2 for i in range(k, n))
            if v_norm < 1e-15:
                continue

            H = [[1.0 if i==j else 0.0 for j in range(n)] for i in range(n)]
            for i in range(n):
                for j in range(n):
                    H[i][j] -= 2*v[i]*v[j]/v_norm

            R_new = [[sum(H[i][k]*R[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
            R = R_new
            Q_new = [[sum(Q[i][k]*H[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
            Q = Q_new

        A_next = [[sum(R[i][k]*Q[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
        A_k = A_next

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

        if all(abs(eigenvalues[i]-eigenvalues_prev[i]) < epsilon for i in range(n)):
            break
        eigenvalues_prev = eigenvalues

    return eigenvalues, iterations, A_k


if __name__ == "__main__":
    A = [
        [3, -5, -4, 7, -1],
        [-1, 17, 1, 2, 2],
        [-2, 3, 4, -1, 5],
        [2, -1, -4, 1, 3],
        [1, 3, -5, 1, 2]
    ]
    EPS = 1e-7
    eigenvalues, iterations, final_matrix = qr_eigenvalues(A, EPS)

    print("Собственные значения матрицы:")
    for idx, eig in enumerate(eigenvalues):
        print(f"λ_{idx+1} = {eig}")

    print(f"\nКоличество итераций: {iterations}")

    print("\nИтоговый вид матрицы:")
    for row in final_matrix:
        print(" | ".join(f"{val:10.6f}" for val in row))
