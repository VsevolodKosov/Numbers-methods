import math


def print_matrix(matrix):
    """Печать матрицы"""
    n = len(matrix[0])
    for row in matrix:
        print('-' * (n * 13))
        print(" | ".join(f"{val:10.6f}" for val in row))
    print('-' * (n * 13))

def householder_reflection(A):
    n = len(A)
    Q = identity_matrix(n)
    
    for k in range(n - 1):
        # Вычисляем вектор v для преобразования Хаусхолдера
        v = [0.0] * n
        norm = 0.0
        
        for i in range(k, n):
            norm += A[i][k] ** 2
        norm = math.sqrt(norm)
        
        sign = 1.0 if A[k][k] >= 0 else -1.0
        v[k] = A[k][k] + sign * norm
        
        for i in range(k + 1, n):
            v[i] = A[i][k]
        
        # Вычисляем норму вектора v
        v_norm = 0.0
        for i in range(k, n):
            v_norm += v[i] ** 2
        
        if v_norm < 1e-15:
            continue
        
        # Создаем матрицу Хаусхолдера H = I - 2*v*v^T/(v^T*v)
        H = identity_matrix(n)
        for i in range(n):
            for j in range(n):
                H[i][j] -= 2 * v[i] * v[j] / v_norm
        
        # Применяем преобразование
        A = matrix_multiply(H, A)
        Q = matrix_multiply(Q, H)
    
    return Q, A

def qr_algorithm(A, epsilon=1e-6):
    n = len(A)
    A_k = [row[:] for row in A]
    eigenvalues_prev = [float('inf')] * n
    eigenvalues = []
    iterations = 0

    while True:
        # QR-разложение
        iterations += 1
        Q, A = householder_reflection(A_k)
        A_k = matrix_multiply(A, Q)

        eigenvalues = []
        i = 0
        while i < n:
            if i < n - 1 and abs(A_k[i + 1][i]) > epsilon:
                # Блок 2x2 для комплексных собственных значений
                a = A_k[i][i]
                b = A_k[i][i + 1]
                c = A_k[i + 1][i]
                d = A_k[i + 1][i + 1]
                
                # Решаем характеристическое уравнение
                trace = a + d
                det = a * d - b * c
                
                discriminant = trace ** 2 - 4 * det
                if discriminant < 0:
                    # Комплексные собственные значения
                    real_part = trace / 2
                    imag_part = math.sqrt(-discriminant) / 2
                    eigenvalues.append(complex(real_part, imag_part))
                    eigenvalues.append(complex(real_part, -imag_part))
                else:
                    # Вещественные собственные значения
                    eigenvalues.append((trace + math.sqrt(discriminant)) / 2)
                    eigenvalues.append((trace - math.sqrt(discriminant)) / 2)
                
                i += 2
            else:
                # Вещественное собственное значение
                eigenvalues.append(A_k[i][i])
                i += 1
        for i in range(n):
            if abs(eigenvalues[i] - eigenvalues_prev[i]) >= epsilon:
                eigenvalues_prev = eigenvalues
                break
        else:
            break

    return eigenvalues, iterations

# Вспомогательные функции для работы с матрицами
def identity_matrix(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

def matrix_multiply(A, B):
    n = len(A)
    m = len(B[0])
    p = len(B)
    
    result = [[0.0 for _ in range(m)] for _ in range(n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                result[i][j] += A[i][k] * B[k][j]
    
    return result

def main():
    A = [
        [3, -5, -4, 7, -1],
        [-1, 17, 1, 2, 2],
        [-2, 3, 4, -1, 5],
        [2, -1, -4, 1, 3],
        [1, 3, -5, 1, 2]
    ]
    EPS = 0.0000001
    eigenvalues, iterations = qr_algorithm(A, EPS)
    
    print("Собственные значения матрицы:")
    for i, eig in enumerate(eigenvalues):
        print(f"λ_{i+1} = {eig}")

    print(f'\nКоличество итераций: {iterations}')
        # Проверка корректности найденных собственных значений
    print("\nПроверка корректности собственных значений:")
    for i, eig in enumerate(eigenvalues):
        # Создаем матрицу (A - λI)
        n = len(A)
        A_minus_lambda_I = [[A[j][k] - (eig if j == k else 0) for k in range(n)] for j in range(n)]
        
        # Вычисляем определитель для проверки (характеристический полином)
        # Используем метод Гаусса для приведения к треугольному виду
        det = 1.0
        temp_matrix = [row[:] for row in A_minus_lambda_I]
        
        for col in range(n):
            # Поиск максимального элемента в столбце
            max_row = col
            for row in range(col + 1, n):
                if abs(temp_matrix[row][col]) > abs(temp_matrix[max_row][col]):
                    max_row = row
            
            # Перестановка строк
            if max_row != col:
                temp_matrix[col], temp_matrix[max_row] = temp_matrix[max_row], temp_matrix[col]
                det *= -1
            
            # Проверка на нулевой столбец
            if abs(temp_matrix[col][col]) < 1e-10:
                det = 0.0
                break
            
            # Умножение определителя на диагональный элемент
            det *= temp_matrix[col][col]
            
            # Обнуление элементов ниже диагонали
            for row in range(col + 1, n):
                factor = temp_matrix[row][col] / temp_matrix[col][col]
                for j in range(col, n):
                    temp_matrix[row][j] -= factor * temp_matrix[col][j]
        
        error = abs(det)
        print(f"|det(A - λ_{i+1}I)| = {error:.2f} {'OK' if error < 0.2 else 'WRONG'}")


if __name__ == "__main__":
    main()