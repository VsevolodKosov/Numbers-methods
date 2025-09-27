
"""
Задание 3: Метод прогонки для решения СЛАУ с трехдиагональной матрицей
Вычисление определителя матрицы
"""

def tridiagonal_matrix_algorithm(a, b, c, d):
    """
    Метод прогонки (алгоритм Томаса) для решения системы с трехдиагональной матрицей
    a - поддиагональ 
    b - главная диагональ
    c - наддиагональ 
    d - правая часть системы
    """
    n = len(d)

    p = [0.0] * n  # прогоночные коэффициенты
    q = [0.0] * n  # прогоночные коэффициенты

    # Прямой ход
    if abs(b[0]) < 1e-15:
        raise ValueError("Нулевой элемент на диагонали b[0]")

    p[0] = -c[0] / b[0]
    q[0] = d[0] / b[0]

    for i in range(1, n):
        denom = b[i] + a[i] * p[i-1]
        if abs(denom) < 1e-15:
            raise ValueError(f"Нулевой знаменатель в строке {i+1}")

        if i < n-1:
            p[i] = -c[i] / denom
        else:
            p[i] = 0.0  # для последнего элемента

        q[i] = (d[i] - a[i] * q[i-1]) / denom

    print("\nОбратный ход (нахождение решения):")
    
    # Обратный ход
    x = [0.0] * n
    x[n-1] = q[n-1]
    print(f"x[{n-1}] = q[{n-1}] = {x[n-1]:.6f}")

    for i in range(n-2, -1, -1):
        x[i] = p[i] * x[i+1] + q[i]
        print(f"x[{i}] = p[{i}]*x[{i+1}] + q[{i}] = {p[i]:.6f}*{x[i+1]:.6f} + {q[i]:.6f} = {x[i]:.6f}")

    print()
    return x


def determinant_tridiagonal(a, b, c):
    """
    Вычисление определителя трехдиагональной матрицы
    """
    n = len(b)
    if n == 1:
        return b[0]
    if n == 2:
        return b[0] * b[1] - a[1] * c[0]

    det_prev_prev = 1.0
    det_prev = b[0]

    for i in range(2, n+1):
        det_current = b[i-1] * det_prev - a[i-1] * c[i-2] * det_prev_prev
        det_prev_prev = det_prev
        det_prev = det_current

    return det_prev


def parse_tridiagonal_equations(equations):
    """
    Парсинг трехдиагональной системы уравнений
    Возвращает коэффициенты a, b, c, d для метода прогонки
    """
    n = len(equations)
    a = [0.0] * n
    b = [0.0] * n
    c = [0.0] * n
    d = [0.0] * n

    for i, eq in enumerate(equations):
        left, right = eq.split('=')
        d[i] = float(right.strip())

        import re
        terms = re.findall(r'([+-]?\s*\d*\.?\d*)\s*x(\d+)', left)

        for coeff_str, var_str in terms:
            var_num = int(var_str) - 1
            
            # Убираем все пробелы из коэффициента
            coeff_str = coeff_str.replace(' ', '')

            if coeff_str == '' or coeff_str == '+':
                coeff = 1.0
            elif coeff_str == '-':
                coeff = -1.0
            else:
                coeff = float(coeff_str)

            if var_num == i - 1:
                a[i] = coeff
            elif var_num == i:
                b[i] = coeff
            elif var_num == i + 1:
                c[i] = coeff

    return a, b, c, d




def check_solution(equations, x):
    """
    Проверка найденного решения подстановкой в исходные уравнения
    """
    print("\nПроверка решения:")
    a, b, c, d = parse_tridiagonal_equations(equations)
    
    for i, eq in enumerate(equations):
        # Вычисляем левую часть уравнения
        left_value = 0.0
        if i > 0:
            left_value += a[i] * x[i-1]  # коэффициент при x[i-1]
        left_value += b[i] * x[i]        # коэффициент при x[i]
        if i < len(x) - 1:
            left_value += c[i] * x[i+1]  # коэффициент при x[i+1]
        
        right_value = d[i]  # правая часть
        
        print(f"Уравнение {i+1}: {left_value:.6f} = {right_value:.6f}")


def solve_tridiagonal_from_equations(equations):
    """
    Решение трехдиагональной системы уравнений методом прогонки
    """
    print("=== Решение трехдиагональной системы методом прогонки ===")
    for eq in equations:
        print(eq)
    print()

    a, b, c, d = parse_tridiagonal_equations(equations)

    # Решение
    x = tridiagonal_matrix_algorithm(a, b, c, d)
    print("Решение системы:")
    for i, xi in enumerate(x, 1):
        print(f"x{i} = {xi:.6f}")

    # Определитель
    det = determinant_tridiagonal(a, b, c)
    print(f"\nОпределитель матрицы: {det:.6f}")
    
    # Проверка решения
    check_solution(equations, x)


def test_tridiagonal_method():
    equations = [
        "7x1 - 3x2 = 26",
        "3x1 + 5x2 - 2x3 = 28",
        "2x2 + 9x3 - x4 = 15",
        "-2x3 + 7x4 - 3x5 = 7",
        "3x4 + 8x5 + x6 = -23",
        "-5x5 + 9x6 + 4x7 = 24",
        "3x6 - 6x7 - 2x8 = -3",
        "3x7 + 8x8 = 24"
    ]
    solve_tridiagonal_from_equations(equations)


if __name__ == "__main__":
    test_tridiagonal_method()
