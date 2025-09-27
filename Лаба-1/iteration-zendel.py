
"""
Задание 4: Метод простых итераций для решения СЛАУ
Решение с точностью ε = 0.0001
"""

def check_convergence_condition(A):
    """
    Проверка достаточного условия сходимости метода простых итераций
    Проверяем диагональное доминирование: |a_ii| > sum(|a_ij|) для j != i
    """
    n = len(A)
    is_diagonally_dominant = True
    weak_rows = []
    
    print("Проверка условия сходимости (диагональное доминирование):")
    for i in range(n):
        diagonal = abs(A[i][i])
        off_diagonal_sum = sum(abs(A[i][j]) for j in range(n) if j != i)
        
        print(f"Строка {i+1}: |a_{i+1}{i+1}| = {diagonal:.6f}, Σ|a_{i+1}j| = {off_diagonal_sum:.6f}")
        
        if diagonal <= off_diagonal_sum:
            is_diagonally_dominant = False
            weak_rows.append(i+1)
            print(f"  Слабое доминирование в строке {i+1}")
        else:
            print(f"  Строгое доминирование в строке {i+1}")
    
    if is_diagonally_dominant:
        print(" Матрица имеет строгое диагональное доминирование - сходимость гарантирована")
    else:
        print(f"   Матрица не имеет строгого диагонального доминирования")
        print(f"   Слабые строки: {weak_rows}")
        print("   Сходимость не гарантирована, но метод может сойтись")
    
    print()
    return is_diagonally_dominant


def simple_iteration(A, b, epsilon=1e-4, max_iterations=10000):
    """
    Метод простых итераций для решения системы Ax = b
    Преобразуем систему к виду x = Bx + c
    """
    n = len(A)
    x = [0.0] * n  # начальное приближение
    
    # Проверка условия сходимости
    check_convergence_condition(A)
    
    # Формируем матрицу B и вектор c для итерационного процесса
    B = [[0.0] * n for _ in range(n)]
    c = [0.0] * n
    
    for i in range(n):
        if abs(A[i][i]) < 1e-15:
            raise ValueError("Нулевой элемент на диагонали - метод не применим")
        
        c[i] = b[i] / A[i][i]
        for j in range(n):
            if i != j:
                B[i][j] = -A[i][j] / A[i][i]
    
    # Итерационный процесс
    for iteration in range(max_iterations):
        x_new = [sum(B[i][j] * x[j] for j in range(n)) + c[i] for i in range(n)]
        
        # Проверка сходимости
        max_diff = max(abs(x_new[i] - x[i]) for i in range(n))
        if max_diff < epsilon:
            print(f"Сходимость достигнута на итерации {iteration + 1}")
            print(f"Максимальная разность: {max_diff:.2e} < ε = {epsilon:.2e}")
            return x_new, iteration + 1
        
        x = x_new
    
    raise ValueError(f"Метод не сошелся за {max_iterations} итераций. Последняя разность: {max_diff:.2e}")

def seidel_method(A, b, epsilon=1e-4, max_iterations=10000):
    """
    Метод Зейделя для решения системы Ax = b
    Использует уже вычисленные значения на текущей итерации
    """
    n = len(A)
    x = [0.0] * n  # начальное приближение
    
    # Проверка условия сходимости
    check_convergence_condition(A)
    
    for iteration in range(max_iterations):
        x_new = x[:]  # копируем текущее приближение
        
        for i in range(n):
            # Сумма с уже обновленными значениями (метод Зейделя)
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            # Сумма со старыми значениями
            s2 = sum(A[i][j] * x[j] for j in range(i+1, n))
            
            if abs(A[i][i]) < 1e-15:
                raise ValueError("Нулевой элемент на диагонали - метод не применим")
            
            x_new[i] = (b[i] - s1 - s2) / A[i][i]
        
        # Проверка сходимости
        max_diff = max(abs(x_new[i] - x[i]) for i in range(n))
        if max_diff < epsilon:
            print(f"Сходимость достигнута на итерации {iteration + 1}")
            print(f"Максимальная разность: {max_diff:.2e} < ε = {epsilon:.2e}")
            return x_new, iteration + 1
        
        x = x_new
    
    raise ValueError(f"Метод не сошелся за {max_iterations} итераций. Последняя разность: {max_diff:.2e}")



def print_matrix(M):
    """Вывод матрицы в удобном формате"""
    for row in M:
        print("  ".join(f"{v:12.6f}" for v in row))


def test_simple_iteration():
    """
    Тестирование метода простых итераций на данных из задания
    """
    print("=== Задание 4: Метод простых итераций ===")
    print("Система уравнений:")
    print("-7x1 + -12x2 + 4x4 = -119")
    print("3x1 + 4x2 + 13x3 - 2x4 = 162")
    print("-4x1 + 3x2 - 8x3 + 17x4 = -125")
    print("13x1 + 3x2 + 5x3 + x4 = 125")
    print()



    # Переупорядочиваем систему для улучшения диагонального доминирования
    A = [
    [13,  3,  5,  1],   # наибольший диагональный элемент для столбца 1
    [-7, -12, 0,  4],   # наибольший по столбцу 2
    [3,   4, 13, -2],   # наибольший по столбцу 3
    [-4,  3, -8, 17]    # наибольший по столбцу 4
    ]
    
    b = [125, -119, 162, -125]

    try:
        # Решение системы методом простых итераций
        x, iterations = simple_iteration(A, b, epsilon=1e-4)
        
        print("Решение системы:")
        for i, xi in enumerate(x, 1):
            print(f"x{i} = {xi:.6f}")
        print()
        
        print(f"Количество итераций: {iterations}")
        print(f"Точность: ε = 0.0001")
        
        # Проверка решения
        check_solution(A, b, x)
        
    except ValueError as e:
        print(f"Ошибка: {e}")


def test_seidel_method():
    """
    Тестирование метода Зейделя на данных из задания
    """
    print("=== Задание 5: Метод Зейделя ===")

    # Переупорядоченная система для улучшения диагонального доминирования
    A = [
    [13,  3,  5,  1],   # наибольший диагональный элемент для столбца 1
    [-7, -12, 0,  4],   # наибольший по столбцу 2
    [3,   4, 13, -2],   # наибольший по столбцу 3
    [-4,  3, -8, 17]    # наибольший по столбцу 4
    ]
    
    b = [125, -119, 162, -125]

    try:
        # Решение системы методом Зейделя
        x, iterations = seidel_method(A, b, epsilon=1e-4)
        
        print("Решение системы:")
        for i, xi in enumerate(x, 1):
            print(f"x{i} = {xi:.6f}")
        print()
        
        print(f"Количество итераций: {iterations}")
        print(f"Точность: ε = 0.0001")
        print()
        
        check_solution_zed(A, b, x)
    except ValueError as e:
        print(f"Ошибка: {e}")

def check_solution(A, b, x):
    """
    Проверка найденного решения подстановкой в исходные уравнения
    """
    print("\nПроверка решения:")
    n = len(A)
    
    for i in range(n):
        left_value = sum(A[i][j] * x[j] for j in range(n))
        right_value = b[i]
        print(f"Уравнение {i+1}: {left_value:.6f} = {right_value:.6f}")
    print()


def check_solution_zed(A, b, x):
    """
    Проверка найденного решения подстановкой в исходные уравнения
    """
    print("\nПроверка решения:")
    n = len(A)
    
    for i in range(n):
        left_value = sum(A[i][j] * x[j] for j in range(n))
        right_value = b[i]
        print(f"Уравнение {i+1}: {left_value:.6f} = {right_value:.6f}")

if __name__ == "__main__":
    test_simple_iteration()
    test_seidel_method()