import matplotlib.pyplot as plt
import math

def cube_root(x):
    if x < 0:
        return -(-x)**(1/3)
    else:
        return x**(1/3)

def f(x):
    """
    Исходная функция:
    y = 3 * ∛(x²) - (x² + 3)/√x + 2
    """
    if x <= 0:
        return float('inf')
    
    try:
        term1 = 3 * cube_root(x**2)          # 3∛(x²)
        term2 = (x**2 + 3) / math.sqrt(x)    # (x² + 3)/√x
        term3 = 2                            # +2
        
        return term1 - term2 + term3
    except (ValueError, ZeroDivisionError):
        return float('inf')

def f_prime(x):
    """
    Аналитическая первая производная:
    f'(x) = [3 * ∛(x) * (1 - x²) + 4 * x^(3/2)] / [2 * x^(11/6)]
    """
    if x <= 0:
        return float('inf')
    
    try:
        numerator = 3 * cube_root(x) * (1 - x**2) + 4 * x**(3/2)
        denominator = 2 * x**(11/6)
        
        if abs(denominator) < 1e-12:
            return float('inf')
        
        return numerator / denominator
    except (ValueError, ZeroDivisionError):
        return float('inf')

def f_double_prime(x):
    """
    Аналитическая вторая производная:
    f''(x) = -[9 * (x^5)^(1/6) * (x² + 3) + 8 * x²] / [12 * x^(10/3)]
    """
    if x <= 0:
        return float('inf')
    
    try:
        term1 = 9 * (x**5)**(1/6) * (x**2 + 3)  # 9 * ∛√(x^5) * (x² + 3)
        term2 = 8 * x**2
        
        numerator = -(term1 + term2)
        denominator = 12 * x**(10/3)
        
        if abs(denominator) < 1e-12:
            return float('inf')
        
        return numerator / denominator
    except (ValueError, ZeroDivisionError):
        return float('inf')

def f_triple_prime(x):
    """
    Аналитическая третья производная:
    f'''(x) = [(27 * (x^5)^(1/6) + 64) * x² + 405 * (x^5)^(1/6)] / [72 * x^(13/3)]
    """
    if x <= 0:
        return float('inf')
    
    try:
        x5_sixth_root = (x**5)**(1/6)  # ∛√(x^5)
        
        term1 = (27 * x5_sixth_root + 64) * x**2
        term2 = 405 * x5_sixth_root
        
        numerator = term1 + term2
        denominator = 72 * x**(13/3)
        
        if abs(denominator) < 1e-12:
            return float('inf')
        
        return numerator / denominator
    except (ValueError, ZeroDivisionError):
        return float('inf')

def f_fourth_prime(x):
    """
    Аналитическая четвертая производная:
    f''''(x) = [-702 * x^(17/6) + (459 * (x^5)^(1/6) - 896) * x² - 8505 * (x^5)^(1/6)] / [432 * x^(16/3)]
    """
    if x <= 0:
        return float('inf')
    
    try:
        x5_sixth_root = (x**5)**(1/6)
        
        term1 = -702 * x**(17/6)
        term2 = (459 * x5_sixth_root - 896) * x**2
        term3 = -8505 * x5_sixth_root
        
        numerator = term1 + term2 + term3
        denominator = 432 * x**(16/3)
        
        if abs(denominator) < 1e-12:
            return float('inf')
        
        return numerator / denominator
    except (ValueError, ZeroDivisionError):
        return float('inf')

def f_sixth_prime(x):
    """
    Аналитическая шестая производная:
    f⁽⁶⁾(x) = [7 * (-68094 * x^(17/6) + (57159 * (x^5)^(1/6) - 66560) * x² - 1082565 * (x^5)^(1/6))] / [15552 * x^(22/3)]
    """
    if x <= 0:
        return float('inf')
    
    try:
        x5_sixth_root = (x**5)**(1/6)  # ∛√(x^5)
        
        term1 = -68094 * x**(17/6)
        term2 = (57159 * x5_sixth_root - 66560) * x**2
        term3 = -1082565 * x5_sixth_root
        
        numerator = 7 * (term1 + term2 + term3)
        denominator = 15552 * x**(22/3)
        
        if abs(denominator) < 1e-12:
            return float('inf')
        
        return numerator / denominator
    except (ValueError, ZeroDivisionError):
        return float('inf')


def generate_grid(a, b, h):
    x_list = []
    x = a
    while x <= b + 1e-10: 
        x_list.append(x)
        x += h
    return x_list


def compute_function_values(x_list):
    return [f(x) for x in x_list]


# ============================================================================
# СХЕМЫ ЧИСЛЕННОГО ДИФФЕРЕНЦИРОВАНИЯ ПЕРВОГО ПОРЯДКА (y')
# ============================================================================

def scheme1_backward(y, h, i):
    """
    Формула 1: y'_i ≈ (y_i - y_{i-1}) / h
    2 точки, назад, порядок точности O(h)
    """
    if i < 1:
        return None
    return (y[i] - y[i-1]) / h


def scheme2_forward(y, h, i):
    """
    Формула 2: y'_i ≈ (y_{i+1} - y_i) / h
    2 точки, вперед, порядок точности O(h)
    """
    if i >= len(y) - 1:
        return None
    return (y[i+1] - y[i]) / h


def scheme3_central(y, h, i):
    """
    Формула 3: y'_i ≈ (y_{i+1} - y_{i-1}) / (2h)
    2 точки, центральная, порядок точности O(h²)
    """
    if i < 1 or i >= len(y) - 1:
        return None
    return (y[i+1] - y[i-1]) / (2 * h)


def scheme4_forward_3point(y, h, i):
    """
    Формула 4: y'_i ≈ (-3y_i + 4y_{i+1} - y_{i+2}) / (2h)
    3 точки, вперед, порядок точности O(h²)
    """
    if i >= len(y) - 2:
        return None
    return (-3 * y[i] + 4 * y[i+1] - y[i+2]) / (2 * h)


def scheme5_backward_3point(y, h, i):
    """
    Формула 5: y'_i ≈ (3y_i - 4y_{i-1} + y_{i-2}) / (2h)
    3 точки, назад, порядок точности O(h²)
    """
    if i < 2:
        return None
    return (3 * y[i] - 4 * y[i-1] + y[i-2]) / (2 * h)


def scheme9_4point(y, h, i):
    """
    Формула 9: y'_i ≈ (-2y_{i-1} - 3y_i + 6y_{i+1} - y_{i+2}) / (6h)
    4 точки, порядок точности O(h³)
    """
    if i < 1 or i >= len(y) - 2:
        return None
    return (-2 * y[i-1] - 3 * y[i] + 6 * y[i+1] - y[i+2]) / (6 * h)


# ============================================================================
# СХЕМЫ ЧИСЛЕННОГО ДИФФЕРЕНЦИРОВАНИЯ ВТОРОГО ПОРЯДКА (y'')
# ============================================================================

def scheme6_forward_2nd(y, h, i):
    """
    Формула 6: y''_i ≈ (y_i - 2y_{i+1} + y_{i+2}) / h²
    3 точки, вперед, порядок точности O(h)
    """
    if i >= len(y) - 2:
        return None
    return (y[i] - 2 * y[i+1] + y[i+2]) / (h ** 2)


def scheme8_central_3point_2nd(y, h, i):
    """
    Формула 8: y''_i ≈ (y_{i+1} - 2y_i + y_{i-1}) / h²
    3 точки, центральная, порядок точности O(h²)
    """
    if i < 1 or i >= len(y) - 1:
        return None
    return (y[i+1] - 2 * y[i] + y[i-1]) / (h ** 2)


def scheme16_4point_2nd(y, h, i):
    """
    Формула 16: y''_i ≈ (2y_i - 5y_{i+1} + 4y_{i+2} - y_{i+3}) / h²
    4 точки, вперед, порядок точности O(h²)
    """
    if i >= len(y) - 3:
        return None
    return (2 * y[i] - 5 * y[i+1] + 4 * y[i+2] - y[i+3]) / (h ** 2)


def scheme23_5point_2nd(y, h, i):
    """
    Формула 23: y''_i ≈ (-y_{i+2} + 16y_{i+1} - 30y_i + 16y_{i-1} - y_{i-2}) / (12h²)
    5 точек, центральная, порядок точности O(h⁴)
    """
    if i < 2 or i >= len(y) - 2:
        return None
    return (-y[i+2] + 16 * y[i+1] - 30 * y[i] + 16 * y[i-1] - y[i-2]) / (12 * h ** 2)


def compute_derivatives_numerical(x_list, y_list, h):
    """
    Вычисление производных численными методами для всех точек сетки.
    """
    n = len(x_list)
    
    # Первые производные (6 схем)
    first_derivatives = {
        'scheme1': [],
        'scheme2': [],
        'scheme3': [],
        'scheme4': [],
        'scheme5': [],
        'scheme9': [],
    }
    
    for i in range(n):
        first_derivatives['scheme1'].append(scheme1_backward(y_list, h, i))
        first_derivatives['scheme2'].append(scheme2_forward(y_list, h, i))
        first_derivatives['scheme3'].append(scheme3_central(y_list, h, i))
        first_derivatives['scheme4'].append(scheme4_forward_3point(y_list, h, i))
        first_derivatives['scheme5'].append(scheme5_backward_3point(y_list, h, i))
        first_derivatives['scheme9'].append(scheme9_4point(y_list, h, i))
    
    # Вторые производные (4 схемы)
    second_derivatives = {
        'scheme6': [],
        'scheme8': [],
        'scheme16': [],
        'scheme23': [],
    }
    
    for i in range(n):
        second_derivatives['scheme6'].append(scheme6_forward_2nd(y_list, h, i))
        second_derivatives['scheme8'].append(scheme8_central_3point_2nd(y_list, h, i))
        second_derivatives['scheme16'].append(scheme16_4point_2nd(y_list, h, i))
        second_derivatives['scheme23'].append(scheme23_5point_2nd(y_list, h, i))
    
    return first_derivatives, second_derivatives


def compute_derivatives_analytical(x_list):
    """
    Вычисление производных аналитически.
    """
    first_deriv = [f_prime(x) for x in x_list]
    second_deriv = [f_double_prime(x) for x in x_list]
    return first_deriv, second_deriv


def compute_theoretical_errors_first_deriv(x_list, h):
    """
    Вычисление теоретических погрешностей для первых производных согласно методичке
    с использованием аналитических производных
    """
    errors = {
        'scheme1': [],
        'scheme2': [],
        'scheme3': [],
        'scheme4': [],
        'scheme5': [],
        'scheme9': [],
    }
    
    for x in x_list:
         # Для схем первого порядка O(h) - вторая производная
        f_double_prime_val = min(abs(f_double_prime(x)), 1000)
            
        # Для схем второго порядка O(h²) - третья производная
        f_triple_prime_val = min(abs(f_triple_prime(x)), 10000)
            
        # Для схем третьего порядка O(h³) - четвертая производная
        f_fourth_prime_val = min(abs(f_fourth_prime(x)), 100000)
            
        # Схема 1 и 2: O(h) - |R(x)| ≤ (h/2)|f''(ξ)|
        error_O_h = (h / 2) * f_double_prime_val
        errors['scheme1'].append(error_O_h)
        errors['scheme2'].append(error_O_h)
            
        # Схема 3: O(h²) - |R(x)| ≤ (h²/6)|f''(ξ)|
        error_O_h2 = (h**2 / 6) * f_double_prime_val
        errors['scheme3'].append(error_O_h2)
            
        # Схема 4 и 5: O(h²) - |R(x)| ≤ (h²/3)|f'''(ξ)|
        error_O_h2_3point = (h**2 / 3) * f_triple_prime_val
        errors['scheme4'].append(error_O_h2_3point)
        errors['scheme5'].append(error_O_h2_3point)
            
        # Схема 9: O(h³) - |R(x)| ≤ (h³/12)|f⁽⁴⁾(ξ)|
        error_O_h3 = (h**3 / 12) * f_fourth_prime_val
        errors['scheme9'].append(error_O_h3)
    
    return errors


def compute_theoretical_errors_second_deriv(x_list, h):
    """
    Вычисление теоретических погрешностей для вторых производных согласно методичке
    с использованием аналитических производных
    """
    errors = {
        'scheme6': [],
        'scheme8': [],
        'scheme16': [],
        'scheme23': [],
    }
    
    for i, x in enumerate(x_list):
        # Используем аналитические производные
        f_triple_prime_val = min(abs(f_triple_prime(x)), 1000)
        f_fourth_prime_val = min(abs(f_fourth_prime(x)), 10000)
        f_sixth_prime_val = min(abs(f_sixth_prime(x)), 100000)
            
        # Схема 6: O(h) - |R(x)| ≤ h|f'''(ξ)|
        errors['scheme6'].append(h * f_triple_prime_val)
            
        # Схема 8: O(h²) - |R(x)| ≤ (h²/12)|f⁽⁴⁾(ξ)|
        errors['scheme8'].append((h**2 / 12) * f_fourth_prime_val)
            
        # Схема 16: O(h²) - |R(x)| ≤ (11/12)h²|f⁽⁴⁾(ξ)|
        errors['scheme16'].append((11/12) * h**2 * f_fourth_prime_val)
            
        # Схема 23: O(h⁴) - |R(x)| ≤ (h⁴/90)|f⁽⁶⁾(ξ)|
        errors['scheme23'].append((h**4 / 90) * f_sixth_prime_val)
            

    return errors


def runge_romberg_refinement(phi_h, phi_kh, k, p):
    """
    Формула Рунге-Ромберга для уточнения результата
    """
    if phi_h is None or phi_kh is None:
        return None
    
    denominator = k**p - 1
    if abs(denominator) < 1e-12:
        return phi_h
    
    return phi_h + (phi_h - phi_kh) / denominator


def compute_runge_romberg(first_deriv_h, first_deriv_h_half, second_deriv_h, second_deriv_h_half, k=2):
    """
    Вычисление уточненных значений по Рунге-Ромбергу для всех схем.
    """
    orders_first = {
        'scheme1': 1, 'scheme2': 1, 'scheme3': 2, 
        'scheme4': 2, 'scheme5': 2, 'scheme9': 3
    }
    
    orders_second = {
        'scheme6': 1, 'scheme8': 2, 'scheme16': 2, 'scheme23': 4
    }
    
    # Уточненные первые производные
    first_deriv_refined = {}
    for scheme in first_deriv_h.keys():
        refined_list = []
        for i in range(len(first_deriv_h[scheme])):
            phi_h = first_deriv_h[scheme][i]
            if i * 2 < len(first_deriv_h_half[scheme]):
                phi_kh = first_deriv_h_half[scheme][i * 2]
            else:
                phi_kh = None
            
            if phi_h is not None and phi_kh is not None:
                refined = runge_romberg_refinement(phi_h, phi_kh, k, orders_first[scheme])
            else:
                refined = phi_h
            refined_list.append(refined)
        first_deriv_refined[scheme] = refined_list
    
    # Уточненные вторые производные
    second_deriv_refined = {}
    for scheme in second_deriv_h.keys():
        refined_list = []
        for i in range(len(second_deriv_h[scheme])):
            phi_h = second_deriv_h[scheme][i]
            if i * 2 < len(second_deriv_h_half[scheme]):
                phi_kh = second_deriv_h_half[scheme][i * 2]
            else:
                phi_kh = None
            
            if phi_h is not None and phi_kh is not None:
                refined = runge_romberg_refinement(phi_h, phi_kh, k, orders_second[scheme])
            else:
                refined = phi_h
            refined_list.append(refined)
        second_deriv_refined[scheme] = refined_list
    
    return first_deriv_refined, second_deriv_refined


def compute_actual_errors(numerical_values, analytical_values):
    """
    Вычисление фактических погрешностей
    """
    errors = {}
    for scheme, values in numerical_values.items():
        error_list = []
        for i, num_val in enumerate(values):
            if num_val is not None and analytical_values[i] is not None:
                error = abs(num_val - analytical_values[i])
            else:
                error = None
            error_list.append(error)
        errors[scheme] = error_list
    return errors


def linspace(start, stop, num):
    """
    Генерация равномерно распределенных точек для построения графиков.
    """
    if num == 1:
        return [start]
    step = (stop - start) / (num - 1)
    return [start + i * step for i in range(num)]


def main():
    # Входные данные
    a = 0.5
    b = 3.5
    h = 0.15
    h_half = h / 2
    k = 2
    
    print("=" * 70)
    print("ЧИСЛЕННОЕ ДИФФЕРЕНЦИРОВАНИЕ")
    print("=" * 70)
    print(f"\nФункция: y = 3 * ∛(x²) - (x² + 3)/√x + 2")
    print(f"Интервал: [{a}, {b}]")
    print(f"Шаг h = {h}")
    print(f"Шаг h/2 = {h_half}")
    
    # Генерация сеток
    x_grid_h = generate_grid(a, b, h)
    x_grid_h_half = generate_grid(a, b, h_half)
    
    # Вычисление значений функции
    y_grid_h = compute_function_values(x_grid_h)
    y_grid_h_half = compute_function_values(x_grid_h_half)
    
    # Аналитические производные
    y_prime_analytical_h, y_double_prime_analytical_h = compute_derivatives_analytical(x_grid_h)
    y_prime_analytical_h_half, y_double_prime_analytical_h_half = compute_derivatives_analytical(x_grid_h_half)
    
    # Численные производные
    first_deriv_h, second_deriv_h = compute_derivatives_numerical(x_grid_h, y_grid_h, h)
    first_deriv_h_half, second_deriv_h_half = compute_derivatives_numerical(x_grid_h_half, y_grid_h_half, h_half)
    
    # Уточнение по Рунге-Ромбергу
    first_deriv_refined, second_deriv_refined = compute_runge_romberg(
        first_deriv_h, first_deriv_h_half, second_deriv_h, second_deriv_h_half, k)
    
    # Вычисление погрешностей для уточненных значений
    first_deriv_errors_actual_refined = compute_actual_errors(first_deriv_refined, y_prime_analytical_h)
    second_deriv_errors_actual_refined = compute_actual_errors(second_deriv_refined, y_double_prime_analytical_h)
    
    first_deriv_errors_theoretical = compute_theoretical_errors_first_deriv(x_grid_h, h)
    second_deriv_errors_theoretical = compute_theoretical_errors_second_deriv(x_grid_h, h)
    
    scheme_names_first = {
        'scheme1': 'Формула 1 (2 точки, назад, O(h))',
        'scheme2': 'Формула 2 (2 точки, вперед, O(h))',
        'scheme3': 'Формула 3 (2 точки, центральная, O(h²))',
        'scheme4': 'Формула 4 (3 точки, вперед, O(h²))',
        'scheme5': 'Формула 5 (3 точки, назад, O(h²))',
        'scheme9': 'Формула 9 (4 точки, O(h³))',
    }
    
    scheme_names_second = {
        'scheme6': 'Формула 6 (3 точки, вперед, O(h))',
        'scheme8': 'Формула 8 (3 точки, центральная, O(h²))',
        'scheme16': 'Формула 16 (4 точки, вперед, O(h²))',
        'scheme23': 'Формула 23 (5 точек, центральная, O(h⁴))',
    }
    
    # Вывод результатов для первой производной - УТОЧНЕННЫЕ ЗНАЧЕНИЯ
    print(f"\n{'='*70}")
    print("ПЕРВАЯ ПРОИЗВОДНАЯ (y') - УТОЧНЕННЫЕ ЗНАЧЕНИЯ (Рунге-Ромберг)")
    print(f"{'='*70}")
    
    for scheme_key, scheme_name in scheme_names_first.items():
        print(f"\n{scheme_name}")
        print(f"{'i':<4} {'x':<10} {'Аналитич.':<12} {'Уточнённое':<12} {'Факт.погр.':<12} {'Теор.погр.':<12}")
        print("-" * 70)
        
        for i in range(len(x_grid_h)):
            x = x_grid_h[i]
            y_anal = y_prime_analytical_h[i]
            refined_val = first_deriv_refined[scheme_key][i]
            refined_error = first_deriv_errors_actual_refined[scheme_key][i]
            theoretical_error = first_deriv_errors_theoretical[scheme_key][i]
            
            refined_str = f"{refined_val:<12.6f}" if refined_val is not None else "---        "
            refined_error_str = f"{refined_error:<12.6f}" if refined_error is not None else "---        "
            theoretical_str = f"{theoretical_error:<12.6f}" if theoretical_error is not None else "---        "
            
            print(f"{i:<4} {x:<10.3f} {y_anal:<12.6f} {refined_str} {refined_error_str} {theoretical_str}")
    
    # Вывод результатов для второй производной - УТОЧНЕННЫЕ ЗНАЧЕНИЯ
    print(f"\n{'='*70}")
    print("ВТОРАЯ ПРОИЗВОДНАЯ (y'') - УТОЧНЕННЫЕ ЗНАЧЕНИЯ (Рунге-Ромберг)")
    print(f"{'='*70}")
    
    for scheme_key, scheme_name in scheme_names_second.items():
        print(f"\n{scheme_name}")
        print(f"{'i':<4} {'x':<10} {'Аналитич.':<12} {'Уточнённое':<12} {'Факт.погр.':<12} {'Теор.погр.':<12}")
        print("-" * 70)
        
        for i in range(len(x_grid_h)):
            x = x_grid_h[i]
            y_anal = y_double_prime_analytical_h[i]
            refined_val = second_deriv_refined[scheme_key][i]
            refined_error = second_deriv_errors_actual_refined[scheme_key][i]
            theoretical_error = second_deriv_errors_theoretical[scheme_key][i]
            
            refined_str = f"{refined_val:<12.6f}" if refined_val is not None else "---        "
            refined_error_str = f"{refined_error:<12.6f}" if refined_error is not None else "---        "
            theoretical_str = f"{theoretical_error:<12.6f}" if theoretical_error is not None else "---        "
            
            print(f"{i:<4} {x:<10.3f} {y_anal:<12.6f} {refined_str} {refined_error_str} {theoretical_str}")
    
    # Построение графиков
    x_plot = linspace(a, b, 500)
    y_plot = [f(x) for x in x_plot]
    y_prime_plot = [f_prime(x) for x in x_plot]
    y_double_prime_plot = [f_double_prime(x) for x in x_plot]
    
    plt.figure(figsize=(16, 14))
    
    plt.subplot(3, 1, 1)
    plt.plot(x_plot, y_plot, 'k-', linewidth=2.5, label='y = f(x)', zorder=3, alpha=0.9)
    plt.plot(x_grid_h, y_grid_h, 'ro', markersize=6, label=f'Узлы сетки (h={h})', zorder=4, alpha=0.8)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Исходная функция', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc='best')
    plt.xlim(a - 0.1, b + 0.1)
    
    plt.subplot(3, 1, 2)
    plt.plot(x_plot, y_prime_plot, 'k-', linewidth=3, label='Аналитическая y\'', zorder=3, alpha=0.9)
    
    colors_h = ['r', 'g', 'b', 'm', 'c', 'orange']
    scheme_keys = ['scheme1', 'scheme2', 'scheme3', 'scheme4', 'scheme5', 'scheme9']
    markers_h = ['o', 's', '^', 'v', 'D', 'p']
    
    for idx, key in enumerate(scheme_keys):
        y_num = first_deriv_refined[key]  # Используем уточненные значения
        valid_indices = [i for i in range(len(x_grid_h)) if y_num[i] is not None]
        if valid_indices:
            x_valid = [x_grid_h[i] for i in valid_indices]
            y_valid = [y_num[i] for i in valid_indices]
            plt.plot(x_valid, y_valid, marker=markers_h[idx], color=colors_h[idx], markersize=5, 
                    label=f'{scheme_names_first[key]}', alpha=0.8, linestyle='None', zorder=4)
    
    plt.xlabel('x', fontsize=12)
    plt.ylabel("y'", fontsize=12)
    plt.title('Первая производная: уточненные численные схемы и аналитическая', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, loc='best', ncol=2, framealpha=0.9)
    plt.xlim(a - 0.1, b + 0.1)
    
    plt.subplot(3, 1, 3)
    plt.plot(x_plot, y_double_prime_plot, 'k-', linewidth=3, label='Аналитическая y\'\'', zorder=3, alpha=0.9)
    
    colors_2nd = ['r', 'g', 'b', 'm']
    scheme_keys_2nd = ['scheme6', 'scheme8', 'scheme16', 'scheme23']
    markers_2nd = ['o', 's', '^', 'v']
    
    for idx, key in enumerate(scheme_keys_2nd):
        y_num = second_deriv_refined[key]  # Используем уточненные значения
        valid_indices = [i for i in range(len(x_grid_h)) if y_num[i] is not None]
        if valid_indices:
            x_valid = [x_grid_h[i] for i in valid_indices]
            y_valid = [y_num[i] for i in valid_indices]
            plt.plot(x_valid, y_valid, marker=markers_2nd[idx], color=colors_2nd[idx], markersize=5, 
                    label=f'{scheme_names_second[key]}', alpha=0.8, linestyle='None', zorder=4)
    
    plt.xlabel('x', fontsize=12)
    plt.ylabel("y''", fontsize=12)
    plt.title('Вторая производная: уточненные численные схемы и аналитическая', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, loc='best', ncol=2, framealpha=0.9)
    plt.xlim(a - 0.1, b + 0.1)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()