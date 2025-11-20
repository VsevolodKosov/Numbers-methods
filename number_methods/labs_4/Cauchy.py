import math
import matplotlib.pyplot as plt

class ButcherTable:
    def __init__(self, A, b, c, name="", implicit=False):
        self.A = A
        self.b = b
        self.c = c
        self.name = name
        self.implicit = implicit
        self.stages = len(c)

def runge_kutta_explicit(f, t_span, y0, h, table):
    """
    Явный метод Рунге-Кутты
    """
    A, b, c = table.A, table.b, table.c
    stages = table.stages
    
    t_start, t_end = t_span
    n_steps = int((t_end - t_start) / h) + 1
    t = [t_start + i * h for i in range(n_steps)]
    y = [0.0] * n_steps
    y[0] = y0

    for n in range(n_steps - 1):
        k = [0.0] * stages
        for i in range(stages):
            sum_terms = 0.0
            for j in range(i):
                sum_terms += A[i][j] * k[j]
            
            t_stage = t[n] + c[i] * h
            y_stage = y[n] + h * sum_terms
            k[i] = f(t_stage, y_stage)
        
        sum_bk = 0.0
        for i in range(stages):
            sum_bk += b[i] * k[i]
        y[n + 1] = y[n] + h * sum_bk
    
    return t, y

def runge_kutta_implicit(f, t_span, y0, h, table, max_iter=100, tol=1e-8):
    """
    Неявный метод Рунге-Кутты с итерационным решением
    """
    A, b, c = table.A, table.b, table.c
    stages = table.stages
    
    t_start, t_end = t_span
    n_steps = int((t_end - t_start) / h) + 1
    t = [t_start + i * h for i in range(n_steps)]
    y = [0.0] * n_steps
    y[0] = y0

    for n in range(n_steps - 1):
        k_prev = [0.0] * stages
        for i in range(stages):
            k_prev[i] = f(t[n], y[n])
        
        for iteration in range(max_iter):
            k_new = [0.0] * stages
            
            for i in range(stages):
                sum_terms = 0.0
                for j in range(stages):
                    sum_terms += A[i][j] * k_prev[j]
                
                t_stage = t[n] + c[i] * h
                y_stage = y[n] + h * sum_terms
                k_new[i] = f(t_stage, y_stage)
            
            max_diff = 0.0
            for i in range(stages):
                diff = abs(k_new[i] - k_prev[i])
                if diff > max_diff:
                    max_diff = diff
            
            if max_diff < tol:
                k_prev = k_new
                break
            
            k_prev = k_new
        
        sum_bk = 0.0
        for i in range(stages):
            sum_bk += b[i] * k_prev[i]
        y[n + 1] = y[n] + h * sum_bk
    
    return t, y

def solve_ode(f, t_span, y0, h, table):
    if table.implicit:
        return runge_kutta_implicit(f, t_span, y0, h, table)
    else:
        return runge_kutta_explicit(f, t_span, y0, h, table)

def calculate_errors(numerical, exact):
    errors = [abs(numerical[i] - exact[i]) for i in range(len(numerical))]
    max_error = max(errors)
    rms_error = math.sqrt(sum(e*e for e in errors) / len(errors))
    return errors, max_error, rms_error

# ЗАДАЧА КОШИ: (x^2) * y' - (x - 1) * y = 0, y(0.5) = (e^2)/2
def f(x, y):
    return ((x - 1) / (x * x)) * y

# Аналитическое решение: y = x * (e^(1/x))
def exact(x):
    return x * math.exp(1 / x)

def format_error(error):
    """Форматирование погрешности в нормальном виде"""
    if error == 0:
        return "0.00000000"
    elif error < 0.000001:
        return f"{error:.10f}"
    else:
        return f"{error:.8f}"

def print_all_values(method_name, x_values, y_numerical, y_exact, errors):
    """Вывод всех значений для одного метода"""
    print(f"\n{method_name}:")
    print("x        Численное    Аналитическое  Погрешность")
    print("-" * 55)
    for i in range(len(x_values)):
        error_str = format_error(errors[i])
        print(f"{x_values[i]:.1f}     {y_numerical[i]:.8f}  {y_exact[i]:.8f}  {error_str}")
    
    max_error = max(errors)
    rms_error = math.sqrt(sum(e*e for e in errors) / len(errors))
    
    max_error_str = format_error(max_error)
    rms_error_str = format_error(rms_error)
    
    print(f"\nМаксимальная погрешность: {max_error_str}")
    print(f"СКЗ погрешность: {rms_error_str}")

def adams_moulton_2(f, x_span, y0, h, max_iter=10, tol=1e-8):
    """
    Метод Адамса-Моултона 2-го порядка (неявный)
    Формула: y_{n+1} = y_n + h(0.5f_{n+1} + 0.5f_n)
    """
    x_start, x_end = x_span
    n_steps = int((x_end - x_start) / h) + 1
    x = [x_start + i * h for i in range(n_steps)]
    y = [0.0] * n_steps
    y[0] = y0

    if n_steps > 1:
        k1 = f(x[0], y[0])
        k2 = f(x[0] + h/2, y[0] + h*k1/2)
        k3 = f(x[0] + h/2, y[0] + h*k2/2)
        k4 = f(x[0] + h, y[0] + h*k3)
        y[1] = y[0] + h * (k1 + 2*k2 + 2*k3 + k4) / 6

    for n in range(1, n_steps - 1):
        f_n = f(x[n], y[n])
        y_pred = y[n] + h * f_n
        
        for iteration in range(max_iter):
            f_next = f(x[n + 1], y_pred)
            y_new = y[n] + h * (0.5 * f_next + 0.5 * f_n)
            
            if abs(y_new - y_pred) < tol:
                y_pred = y_new
                break
            y_pred = y_new
        
        y[n + 1] = y_pred
    
    return x, y

# ТАБЛИЦЫ БУТЧЕРА

# Явный метод Эйлера (1-й порядок)
euler = ButcherTable(
    A=[[0]],
    b=[1],
    c=[0],
    name="Явный Эйлер (1-й порядок)",
    implicit=False
)

# Неявный метод Эйлера (1-й порядок)
implicit_euler = ButcherTable(
    A=[[1]],
    b=[1],
    c=[1],
    name="Неявный Эйлер (1-й порядок)",
    implicit=True
)

# Метод Хойна 2-го порядка
heun_method = ButcherTable(
    A=[[0, 0],
       [1, 0]],
    b=[0.5, 0.5],
    c=[0, 1],
    name="Метод Хойна 2 порядка",
    implicit=False
)

# Метод Ральстона 2-го порядка
ralston_method = ButcherTable(
    A=[[0, 0],
       [2/3, 0]],
    b=[1/4, 3/4],
    c=[0, 2/3],
    name="Метод Ральстона 2 порядка",
    implicit=False
)

# Метод РК3 3-го порядка
rk3_method = ButcherTable(
    A=[[0, 0, 0],
       [1/2, 0, 0],
       [-1, 2, 0]],
    b=[1/6, 2/3, 1/6],
    c=[0, 1/2, 1],
    name="Метод РК3 (3-й порядок)",
    implicit=False
)

def plot_solutions(methods_results, x_exact, y_exact):
    """Построение графика решений"""
    plt.figure(figsize=(15, 10))
    
    plt.plot(x_exact, y_exact, 'black', linewidth=4, label='Аналитическое решение: y = x * e^(1/x)')
    
    colors = ['red', 'green', 'lightblue', 'orange', 'purple', 'brown']
    
    for i, (method_name, x, y) in enumerate(methods_results):
        plt.plot(x, y, color=colors[i], linewidth=2, marker='o', markersize=4, label=method_name)
    
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y(x)', fontsize=14)
    plt.title('Сравнение численных решений с аналитическим решением', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

def main():
    print("Лабораторная работа: Решение задачи Коши")
    print("=" * 70)
    print("ЗАДАЧА КОШИ: (x²)y' - (x - 1)y = 0")
    print("Начальное условие: y(0.5) = e²/2")
    print("Интервал: x ∈ [0.5; 2]")
    print("Аналитическое решение: y(x) = x * e^(1/x)")
    print("=" * 70)
    
    # Параметры задачи
    x_span = (0.5, 2.0)
    h = 0.1
    y0 = math.exp(2) / 2
    
    # Все методы Рунге-Кутты
    rk_methods = [
        euler,
        implicit_euler,
        heun_method,
        ralston_method,
        rk3_method
    ]
    
    # Аналитическое решение для сравнения
    x_exact = [x_span[0] + i * h for i in range(int((x_span[1] - x_span[0]) / h) + 1)]
    y_exact = [exact(x) for x in x_exact]
    
    methods_results = []
    
    # Решение методами Рунге-Кутты
    for method in rk_methods:
        x, y = solve_ode(f, x_span, y0, h, method)
        errors, max_err, rms_err = calculate_errors(y, y_exact)
        print_all_values(method.name, x, y, y_exact, errors)
        methods_results.append((method.name, x, y))
    
    # Решение методом Адамса-Моултона
    x_am, y_am = adams_moulton_2(f, x_span, y0, h)
    errors_am, max_err_am, rms_err_am = calculate_errors(y_am, y_exact)
    print_all_values("Адамс-Моултон 2 порядка (неявный)", x_am, y_am, y_exact, errors_am)
    methods_results.append(("Адамс-Моултон 2 порядка (неявный)", x_am, y_am))
    
    # СВОДНАЯ ТАБЛИЦА ПОГРЕШНОСТЕЙ
    print("\n" + "=" * 70)
    print("СВОДНАЯ ТАБЛИЦА ПОГРЕШНОСТЕЙ")
    print("=" * 70)
    print("Метод                          | Макс. ошибка    | СКЗ ошибка")
    print("-" * 70)
    
    for method in rk_methods:
        x, y = solve_ode(f, x_span, y0, h, method)
        errors, max_err, rms_err = calculate_errors(y, y_exact)
        max_str = format_error(max_err)
        rms_str = format_error(rms_err)
        print(f"{method.name:30} | {max_str:14} | {rms_str:14}")
    
    errors_am, max_err_am, rms_err_am = calculate_errors(y_am, y_exact)
    max_str_am = format_error(max_err_am)
    rms_str_am = format_error(rms_err_am)
    print(f"{'Адамс-Моултон 2 порядка (неявный)':30} | {max_str_am:14} | {rms_str_am:14}")
    
    plot_solutions(methods_results, x_exact, y_exact)

if __name__ == "__main__":
    main()