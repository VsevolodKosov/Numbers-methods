import math
import matplotlib.pyplot as plt
import numpy as np

# ----------------------- Функции системы -----------------------
def f(x1, x2):
    """Исходная система уравнений"""
    return [x1**2 - math.exp(x2) + 1,
            x2**2 - 2*math.sin(x1) - 3]

def J(x1, x2):
    """Матрица Якоби для исходной системы"""
    return [[2*x1, -math.exp(x2)],
            [-2*math.cos(x1), 2*x2]]

def solve_linear_2x2(A, b):
    """Решение системы 2x2 линейных уравнений"""
    a11, a12 = A[0]
    a21, a22 = A[1]
    b1, b2 = b
    det = a11*a22 - a12*a21
    if det == 0:
        raise ValueError("Матрица вырождена")
    x = (b1*a22 - b2*a12) / det
    y = (a11*b2 - a21*b1) / det
    return [x, y]

# ----------------------- Преобразованные функции для итерационных методов -----------------------
def phi1(x2):
    """Преобразованная первая функция: x1 = sqrt(exp(x2) - 1)"""
    # Проверяем область определения
    if math.exp(x2) - 1 < 0:
        raise ValueError(f"phi1 не определена для x2={x2}: exp({x2})-1 < 0")
    return math.sqrt(math.exp(x2) - 1)

def phi2(x1):
    """Преобразованная вторая функция: x2 = sqrt(3 + 2*sin(x1))"""
    # Проверяем область определения
    if 3 + 2*math.sin(x1) < 0:
        raise ValueError(f"phi2 не определена для x1={x1}: 3+2*sin({x1}) < 0")
    return math.sqrt(3 + 2*math.sin(x1))

def J_Phi(x1, x2):
    """Матрица Якоби для преобразованной системы"""
    dphi1_dx1 = 0  
    
    if math.exp(x2) - 1 > 0:
        dphi1_dx2 = (math.exp(x2)) / (2 * math.sqrt(math.exp(x2) - 1))
    else:
        dphi1_dx2 = 0
        
    if 3 + 2*math.sin(x1) > 0:
        dphi2_dx1 = (2 * math.cos(x1)) / (2 * math.sqrt(3 + 2*math.sin(x1)))
    else:
        dphi2_dx1 = 0
        
    dphi2_dx2 = 0 
        
    return [[dphi1_dx1, dphi1_dx2],
            [dphi2_dx1, dphi2_dx2]]

def matrix_norm_1(J_matrix):
    """Первая норма матрицы (максимальная сумма по столбцам)"""
    a11, a12 = J_matrix[0]
    a21, a22 = J_matrix[1]
    return max(abs(a11) + abs(a21), abs(a12) + abs(a22))

def check_convergence_condition(x1, x2):
    """
    Проверяет условие сходимости для метода простых итераций
    через норму матрицы Якоби преобразованной системы
    """
    J_phi = J_Phi(x1, x2)
    norm = matrix_norm_1(J_phi)
    return norm < 1

# ----------------------- МЕТОД ПРОСТЫХ ИТЕРАЦИЙ -----------------------
def simple_iteration_method(x1_0, x2_0, eps):
    """
    Классический метод простых итераций
    x_i^(k+1) = Φ_i(x_1^(k), x_2^(k), ..., x_n^(k)), i = 1, n
    """
    x1, x2 = x1_0, x2_0
    iterations = 0
    
    # Проверяем условие сходимости в начальной точке
    if not check_convergence_condition(x1, x2):
        print("Метод простых итераций: Условие сходимости не выполняется!")
        return [x1, x2], 0, False
    
    while True:
        x1_old, x2_old = x1, x2
        iterations += 1
        
        try:
            # ПАРАЛЛЕЛЬНОЕ обновление (формула 2.69)
            x1_new = phi1(x2_old)
            x2_new = phi2(x1_old)
            
            x1, x2 = x1_new, x2_new
            
            # Проверяем сходимость (формула 2.72)
            max_change = max(abs(x1 - x1_old), abs(x2 - x2_old))
            
            if max_change < eps:
                return [x1, x2], iterations, True
                
        except ValueError as e:
            print(f"Метод простых итераций: Ошибка на итерации {iterations}: {e}")
            return [x1_old, x2_old], iterations, False

# ----------------------- МЕТОД ЗЕЙДЕЛЯ -----------------------
def seidel_method(x1_0, x2_0, eps):
    """
    Классический метод Зейделя
    x_1^(k+1) = Φ_1(x_1^(k), x_2^(k), ..., x_n^(k))
    x_2^(k+1) = Φ_2(x_1^(k+1), x_2^(k), ..., x_n^(k))
    """
    x1, x2 = x1_0, x2_0
    iterations = 0
    
    # Проверяем условие сходимости в начальной точке
    if not check_convergence_condition(x1, x2):
        print("Метод Зейделя: Условие сходимости не выполняется!")
        return [x1, x2], 0, False
    
    while True:
        x1_old, x2_old = x1, x2
        iterations += 1
        
        try:
            # ПОСЛЕДОВАТЕЛЬНОЕ обновление (формула 2.70)
            x1_new = phi1(x2_old)      # x_1^(k+1) = Φ_1(x_2^(k))
            x1 = x1_new
            
            x2_new = phi2(x1)          # x_2^(k+1) = Φ_2(x_1^(k+1))
            x2 = x2_new
            
            # Проверяем сходимость (формула 2.72)
            max_change = max(abs(x1 - x1_old), abs(x2 - x2_old))
            
            if max_change < eps:
                return [x1, x2], iterations, True
                
        except ValueError as e:
            print(f"Метод Зейделя: Ошибка на итерации {iterations}: {e}")
            return [x1_old, x2_old], iterations, False

# ----------------------- МЕТОД НЬЮТОНА (способ 1 - обратная матрица) -----------------------
def newton_method_v1(x1_0, x2_0, eps):
    """
    Первый способ метода Ньютона
    """
    x1, x2 = x1_0, x2_0
    iterations = 0
    
    while True:
        F_val = f(x1, x2)
        J_val = J(x1, x2)
        
        # Вычисляем J^(-1) * f(X) через решение системы
        J_inv_f = solve_linear_2x2(J_val, F_val)
        
        # Новые значения
        x1_new = x1 - J_inv_f[0]
        x2_new = x2 - J_inv_f[1]
        
        iterations += 1
        
        # Условие остановки
        max_change = max(abs(x1_new - x1), abs(x2_new - x2))
        x1, x2 = x1_new, x2_new
        
        if max_change < eps:
            return [x1, x2], iterations, True

# ----------------------- МЕТОД НЬЮТОНА (способ 2 - определители) -----------------------
def newton_method_v2(x1_0, x2_0, eps):
    """
    Второй способ метода Ньютона: через определители по формулам Крамера
    """
    x1, x2 = x1_0, x2_0
    iterations = 0
    
    while True:
        F_val = f(x1, x2)
        J_val = J(x1, x2)
        
        # Элементы матрицы Якоби
        a11, a12 = J_val[0]
        a21, a22 = J_val[1]
        
        # Элементы вектора f(X)
        f1, f2 = F_val
        
        # Определитель матрицы Якоби
        det_J = a11 * a22 - a12 * a21
        
        if det_J == 0:
            return [x1, x2], iterations, False
        
        # Матрицы A1 и A2 для метода Крамера
        det_A1 = f1 * a22 - f2 * a12
        det_A2 = a11 * f2 - a21 * f1
        
        # Вычисляем приращения
        delta_x1 = -det_A1 / det_J
        delta_x2 = -det_A2 / det_J
        
        x1_new = x1 + delta_x1
        x2_new = x2 + delta_x2
        
        iterations += 1
        
        max_change = max(abs(x1_new - x1), abs(x2_new - x2))
        x1, x2 = x1_new, x2_new
        
        if max_change < eps:
            return [x1, x2], iterations, True

# ----------------------- Поиск начальных приближений -----------------------
def find_initial_approximations(step=0.01, threshold=0.01):
    initial_approximations = []

    x2_vals = []
    x = -3.0
    while x <= 3.0:
        x2_vals.append(x)
        x += step

    x1_vals = []
    x = -3.0
    while x <= 3.0:
        x1_vals.append(x)
        x += step

    # Вычисляем возможные x1 по первой функции
    x1_from_x2 = []
    for x2_val in x2_vals:
        val = math.exp(x2_val) - 1
        if val >= 0:
            sqrt_val = math.sqrt(val)
            x1_from_x2.append([sqrt_val, x2_val])
            x1_from_x2.append([-sqrt_val, x2_val])

    # Вычисляем возможные x2 по второй функции
    x2_from_x1 = []
    for x1_val in x1_vals:
        val = 2*math.sin(x1_val) + 3
        if val >= 0:
            sqrt_val = math.sqrt(val)
            x2_from_x1.append([x1_val, sqrt_val])
            x2_from_x1.append([x1_val, -sqrt_val])

    # Ищем пересечения
    for x1a, x2a in x1_from_x2:
        for x1b, x2b in x2_from_x1:
            if abs(x1a - x1b) < threshold and abs(x2a - x2b) < threshold:
                too_close = False
                for approx in initial_approximations:
                    if math.hypot(x1a - approx[0], x2a - approx[1]) < 0.05:
                        too_close = True
                        break
                if not too_close:
                    initial_approximations.append([x1a, x2a])

    return initial_approximations

# ----------------------- Построение графика системы -----------------------
def plot_system(initial_approximations):
    plt.figure(figsize=(12, 10))

    # Создаем сетку для построения
    x1 = np.linspace(-3, 3, 500)
    x2 = np.linspace(-3, 3, 500)
    X1, X2 = np.meshgrid(x1, x2)

    # Вычисляем значения функций
    F1 = X1**2 - np.exp(X2) + 1
    F2 = X2**2 - 2*np.sin(X1) - 3

    # Построение контуров F1 = 0 и F2 = 0
    contour1 = plt.contour(X1, X2, F1, levels=[0], colors='blue', linewidths=2)
    contour2 = plt.contour(X1, X2, F2, levels=[0], colors='red', linewidths=2)

    # Подписи для контуров
    plt.clabel(contour1, inline=True, fontsize=10, fmt='F1=0')
    plt.clabel(contour2, inline=True, fontsize=10, fmt='F2=0')

    # Отмечаем начальные приближения
    if initial_approximations:
        for i, approx in enumerate(initial_approximations):
            plt.plot(approx[0], approx[1], 'go', markersize=10, markeredgecolor='black', 
                    label='Нач. приближение' if i == 0 else "")
            # Подписываем точки с координатами
            plt.annotate(f'({approx[0]:.2f}, {approx[1]:.2f})', 
                        (approx[0], approx[1]), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    # Настройка графика
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Графики системы уравнений: x1² - exp(x2) + 1 = 0 и x2² - 2*sin(x1) - 3 = 0')
    plt.legend()
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)

    plt.tight_layout()
    plt.show()

    return initial_approximations


# ----------------------- Точка входа -----------------------
if __name__ == "__main__":
    eps = 1e-4

    # === ВЫБОР НАЧАЛЬНЫХ ПРИБЛИЖЕНИЙ ===
    selected_approximations = [
         [-2, -1.8],  
         [2, 2.2],
         [2.56, 2.02],  
    ]
    
    auto_approximations = find_initial_approximations()
    
    if not selected_approximations:
        selected_approximations = auto_approximations
    
    print("=" * 70)
    print("Начальные приближения, определенные графически:")
    for i, approx in enumerate(auto_approximations):
        print(f"Приближение {i+1}: [{approx[0]:.3f}, {approx[1]:.3f}]")
    
    print("\n" + "=" * 70)
    print("Начальные приближения (для вычислений):")
    for i, approx in enumerate(selected_approximations):
        print(f"Приближение {i+1}: [{approx[0]:.3f}, {approx[1]:.3f}]")
    print("=" * 70)

    plot_system(auto_approximations)
    
    roots_info = []
    
    colors = ['red', 'purple', 'brown', 'pink']
    
    print("\nРезультаты:")
    
    for i, (x1_0, x2_0) in enumerate(selected_approximations):
        print(f"\n--- Начальное приближение {i+1}: [{x1_0}, {x2_0}] ---")
        
        # Метод Ньютона
        X_newton_v1, iter_newton_v1, conv_newton_v1 = newton_method_v1(x1_0, x2_0, eps)
        X_newton_v2, iter_newton_v2, conv_newton_v2 = newton_method_v2(x1_0, x2_0, eps)
        
        # Метод простых итераций
        X_simple, iter_simple, conv_simple = simple_iteration_method(x1_0, x2_0, eps)
        
        # Метод Зейделя
        X_seidel, iter_seidel, conv_seidel = seidel_method(x1_0, x2_0, eps)
        
        # Вывод результатов
        if conv_newton_v1:
            print(f"Метод Ньютона (способ 1): x = [{X_newton_v1[0]:.6f}, {X_newton_v1[1]:.6f}], итерации = {iter_newton_v1}")
            roots_info.append({'root': X_newton_v1, 'method': 'Ньютон (1)', 'color': colors[0]})
        else:
            print("Метод Ньютона (способ 1): не сошелся")
            
        if conv_newton_v2:
            print(f"Метод Ньютона (способ 2): x = [{X_newton_v2[0]:.6f}, {X_newton_v2[1]:.6f}], итерации = {iter_newton_v2}")
            roots_info.append({'root': X_newton_v2, 'method': 'Ньютон (2)', 'color': colors[1]})
        else:
            print("Метод Ньютона (способ 2): не сошелся")
            
        if conv_simple:
            print(f"Метод простых итераций: x = [{X_simple[0]:.6f}, {X_simple[1]:.6f}], итерации = {iter_simple}")
            roots_info.append({'root': X_simple, 'method': 'Простые итерации', 'color': colors[2]})
        else:
            print("Метод простых итераций: не сошелся")
            
        if conv_seidel:
            print(f"Метод Зейделя: x = [{X_seidel[0]:.6f}, {X_seidel[1]:.6f}], итерации = {iter_seidel}")
            roots_info.append({'root': X_seidel, 'method': 'Зейделя', 'color': colors[3]})
        else:
            print("Метод Зейделя: не сошелся")