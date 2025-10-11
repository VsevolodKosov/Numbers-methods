def simple_iteration(A, b, epsilon=1e-4, max_iterations=10000):
    """
    Метод простых итераций для решения системы Ax = b
    """
    n = len(A)
    x = [0.0] * n  # начальное приближение
    
    # Формируем матрицу B и вектор c
    B = [[0.0] * n for _ in range(n)]
    c = [0.0] * n
    
    for i in range(n):
        c[i] = b[i] / A[i][i]
        for j in range(n):
            if i != j:
                B[i][j] = -A[i][j] / A[i][i]
    
    # Итерационный процесс
    for iteration in range(max_iterations):
        x_new = [sum(B[i][j] * x[j] for j in range(n)) + c[i] for i in range(n)]
        max_diff = max(abs(x_new[i] - x[i]) for i in range(n))
        if max_diff < epsilon:
            return x_new, iteration + 1
        x = x_new
    
    return x, max_iterations
    
def seidel_method(A, b, epsilon=1e-4, max_iterations=10000):
    """
    Метод Зейделя для решения системы Ax = b
    """
    n = len(A)
    x = [0.0] * n  # начальное приближение
    
    for iteration in range(max_iterations):
        x_new = x[:]
        for i in range(n):
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i+1, n))
            x_new[i] = (b[i] - s1 - s2) / A[i][i]
        
        max_diff = max(abs(x_new[i] - x[i]) for i in range(n))
        if max_diff < epsilon:
            return x_new, iteration + 1
        x = x_new
    
    return x, max_iterations

def check_solution(A, b, x):
    """
    Проверка найденного решения подстановкой в исходные уравнения
    """
    print("\nПроверка решения:")
    n = len(A)
    for i in range(n):
        left_value = sum(A[i][j] * x[j] for j in range(n))
        print(f"Уравнение {i+1}: {left_value:.6f} = {b[i]:.6f}")
    print()

if __name__ == "__main__":
    A = [
        [13, 3, 5, 1],
        [-7, -12, 0, 4],
        [3, 4, 13, -2],
        [-4, 3, -8, 17]
    ]
    b = [125, -119, 162, -125]

    # Метод простых итераций
    print("=== Метод простых итераций ===")
    x_iter, iterations_iter = simple_iteration(A, b, epsilon=1e-4)
    print("Решение:")
    for i, xi in enumerate(x_iter, 1):
        print(f"x{i} = {xi:.6f}")
    print(f"Количество итераций: {iterations_iter}")
    check_solution(A, b, x_iter)

    # Метод Зейделя
    print("\n=== Метод Зейделя ===")
    x_seidel, iterations_seidel = seidel_method(A, b, epsilon=1e-4)
    print("Решение:")
    for i, xi in enumerate(x_seidel, 1):
        print(f"x{i} = {xi:.6f}")
    print(f"Количество итераций: {iterations_seidel}")
    check_solution(A, b, x_seidel)
