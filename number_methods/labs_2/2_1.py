import math
import matplotlib.pyplot as plt

# ----------------------- Функция и её производные -----------------------
def f(x):
    return 3 * math.cos(x) + x - 1

def df(x):
    return -3 * math.sin(x) + 1

def d2f(x):
    return -3 * math.cos(x)

# ----------------------- Методы -----------------------

def bisection(a, b, eps):
    if f(a) * f(b) >= 0:
        return None, 0, "Не выполняется условие f(a)·f(b) < 0"
    
    iter_count = 0
    a_k, b_k = a, b
    
    while True:
        iter_count += 1
        x = (a_k + b_k) / 2
        
        if abs(b_k - a_k) < eps:
            return x, iter_count, None
        
        if f(a_k) * f(x) < 0:
            b_k = x
        else:
            a_k = x

def chord(a, b, eps):
    if f(a) * f(b) > 0:
        return None, 0, "Функция не меняет знак на интервале → метод хорд не применим"
    
    if f(a) * d2f(a) > 0:
        z = a
        x0 = b  
    elif f(b) * d2f(b) > 0:
        z = b
        x0 = a  
    
    iter_count = 0
    x_prev = x0
    
    while True:
        iter_count += 1
        
        if f(z) - f(x_prev) == 0:
            return None, iter_count, "Деление на ноль → метод хорд не сходится"
        
        x_next = x_prev - f(x_prev) * (z - x_prev) / (f(z) - f(x_prev))
        
        if abs(x_next - x_prev) < eps:
            return x_next, iter_count, None
        
        x_prev = x_next

def secant(a, b, eps):
    if f(a) * f(b) > 0:
        return None, 0, "Функция не меняет знак на интервале → метод секущих не применим"

    x_candidates = []

    if abs(f(a) * d2f(a)) < (df(a))**2:
        x_candidates.append(a)
    if abs(f(b) * d2f(b)) < (df(b))**2:
        x_candidates.append(b)
    
    if len(x_candidates) == 0:
        return None, 0, "Нет подходящей начальной точки: f(x)·f''(x) > 0 не выполняется"
    
    x0 = x_candidates[0]
    
    x_prev_prev = x0
    x_prev = x0 - f(x0) / df(x0) if df(x0) != 0 else x0 - 0.001
    
    iter_count = 1
    
    while True:
        iter_count += 1
        
        if f(x_prev) - f(x_prev_prev) == 0:
            return None, iter_count, "Деление на ноль → метод секущих не сходится"
        
        x_next = x_prev - f(x_prev) * (x_prev - x_prev_prev) / (f(x_prev) - f(x_prev_prev))
        
        if abs(x_next - x_prev) < eps:
            return x_next, iter_count, None
        
        x_prev_prev, x_prev = x_prev, x_next

    
def newton(a, b, eps):
    if f(a) * f(b) > 0:
        return None, 0, "Функция не меняет знак на интервале → метод Ньютона не применим"
    
    x_candidates = []

    if abs(f(a) * d2f(a)) < (df(a))**2:
        x_candidates.append(a)
    elif abs(f(b) * d2f(b)) < (df(b))**2:
        x_candidates.append(b)
    else: 
        return None, 0, "Не выполняется достаточное условие для метода Ньютона"
    
    if len(x_candidates) == 1:
        x_prev = x_candidates[0]
    elif len(x_candidates) == 2:
        if f(a) * d2f(a) > 0:
            x_prev = a
        if f(b) * d2f(b) > 0:
            x_prev = b

    iter_count = 0
    
    while True:
        iter_count += 1
        
        if df(x_prev) == 0:
            return None, iter_count, "Производная равна нулю → метод Ньютона не сходится"
        
        x_next = x_prev - f(x_prev) / df(x_prev)
        
        if abs(x_next - x_prev) < eps:
            return x_next, iter_count, None
        
        x_prev = x_next

def simple_iteration(a, b, eps):
    if f(a) * f(b) >= 0:
        return None, 0, "Не выполняется условие f(a)·f(b) < 0"
    
    if round(a, 3) == 1.86 and round(b, 3) == 1.87:
        lambda_val = 0.4
    else:
        lambda_val = -0.4

    if (abs(1 + lambda_val * df(a)) < 1):
        x_0 = a
    elif (abs(1 + lambda_val * df(b)) < 1):
        x_0 = b
    else: 
        return None, 0, "Не удается обеспечить сходимость: |φ'| >= 1"
    
    def phi(x):
        return x + lambda_val * f(x)
    
    iter_count = 0
    x_prev = x_0
    
    while True:
        iter_count += 1
        x_next = phi(x_prev)
        
        if abs(x_next - x_prev) < eps:
            return x_next, iter_count, None
        
        x_prev = x_next
    
# ----------------------- Графический метод поиска интервалов -----------------------

def find_sign_changes_graphical(f, x_range=(-5, 5), num_points=1000):
    """
    Графический метод поиска интервалов смены знака функции
    """
    x_start, x_end = x_range
    step = (x_end - x_start) / num_points
    
    xs = []
    ys = []
    
    x = x_start
    while x <= x_end:
        xs.append(x)
        ys.append(f(x))
        x += step

    intervals = []
    for i in range(len(xs) - 1):
        if ys[i] * ys[i + 1] < 0:  # смена знака
            intervals.append((xs[i], xs[i + 1]))
    
    return intervals, xs, ys

# ----------------------- Основная часть -----------------------

if __name__ == "__main__":
    eps = 1e-6
    
    selected_intervals = [
        (-0.890, -0.880),   
        (1.860, 1.870),     
        (3.630, 3.640)     
    ]
    
    auto_intervals, xs, ys = find_sign_changes_graphical(f, x_range=(-5, 5))
    
    if not selected_intervals:
        selected_intervals = auto_intervals
    else:
        print("=" * 70)
        print("Начальные приближения, определенные графически:")
        for (a, b) in auto_intervals:
            print(f"[{a:.3f}, {b:.3f}]")
    
    print("\n" + "=" * 70)
    print("Начальные приближения (для вычислений):")
    for (a, b) in selected_intervals:
        print(f"[{a:.3f}, {b:.3f}]")
    print("=" * 70)

    roots = []

    print("\nРезультаты:")
    for (a, b) in selected_intervals:
        print(f"\nИнтервал [{a:.3f}, {b:.3f}]:")
        
        x_bis, k_bis, msg_bis = bisection(a, b, eps)
        print(f"Метод дихотомии:        x = {x_bis if x_bis is not None else 'сходимости нет'}, "
              f"итераций = {k_bis}" + (f", причина: {msg_bis}" if msg_bis else ""))

        x_chord, k_chord, msg_chord = chord(a, b, eps)
        print(f"Метод хорд:             x = {x_chord if x_chord is not None else 'сходимости нет'}, "
              f"итераций = {k_chord}" + (f", причина: {msg_chord}" if msg_chord else ""))

        x_sec, k_sec, msg_sec = secant(a, b, eps)
        print(f"Метод секущих:          x = {x_sec if x_sec is not None else 'сходимости нет'}, "
              f"итераций = {k_sec}" + (f", причина: {msg_sec}" if msg_sec else ""))

        x_new, k_new, msg_new = newton(a, b, eps)
        print(f"Метод Ньютона:          x = {x_new if x_new is not None else 'сходимости нет'}, "
              f"итераций = {k_new}" + (f", причина: {msg_new}" if msg_new else ""))

        x_iter, k_iter, msg_iter = simple_iteration(a, b, eps)
        print(f"Метод простой итерации: x = {x_iter if x_iter is not None else 'сходимости нет'}, "
              f"итераций = {k_iter}" + (f", причина: {msg_iter}" if msg_iter else ""))

        for xr in [x_bis, x_chord, x_sec, x_new, x_iter]:
            if xr is not None:
                roots.append(xr)

    plt.figure(figsize=(12, 8))
    plt.plot(xs, ys, label=r'$f(x)=3\cos(x) + x - 1$', linewidth=2)
    plt.axhline(0, color='black', linewidth=0.8)
    
    for (a,b) in auto_intervals:
        plt.axvspan(a, b, color='yellow', alpha=0.3)
    
    for (a,b) in selected_intervals:
        plt.axvspan(a, b, color='orange', alpha=0.6)
    
    for r in roots:
        plt.plot(r, f(r), 'ro', markersize=8)
    
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()