import math

def f(x):
    return x / (x**2 + x + 1)

def df(x):
    return (1 - x**2) / (x**2 + x + 1)**2

def d2f(x):
    return (2*x*(x**2 - 3) - 2) / (x**2 + x + 1)**3

def d4f(x):
    return (24*(x**6 - 15*x**4 + 15*x**2 - 1)) / (x**2 + x + 1)**5

def create_uniform_grid(a, b, n):
    h = (b - a) / n
    nodes = [a + i * h for i in range(n + 1)]
    return nodes, h

def find_max_derivative(derivative_func, a, b):
    points = [a, b, (a + b) / 2]
    return max(abs(derivative_func(x)) for x in points)

def calculate_optimal_n(a, b, epsilon=1e-6):
    max_d2f = find_max_derivative(d2f, a, b)
    
    n_optimal = int((b - a) * math.sqrt(max_d2f / (24 * epsilon))) + 1
    
    n_optimal = max(4, n_optimal)
    if n_optimal % 2 != 0:
        n_optimal += 1
    
    return n_optimal

def midpoint_rule(a, b, n):
    nodes, h = create_uniform_grid(a, b, n)
    integral = 0
    for i in range(n):
        x_mid = (nodes[i] + nodes[i+1]) / 2
        integral += f(x_mid)
    return h * integral

def trapezoidal_rule(a, b, n):
    nodes, h = create_uniform_grid(a, b, n)
    integral = (f(nodes[0]) + f(nodes[-1])) / 2
    for i in range(1, n):
        integral += f(nodes[i])
    return h * integral

def simpson_rule(a, b, n):
    if n % 2 != 0:
        n += 1
    nodes, h = create_uniform_grid(a, b, n)
    integral = f(nodes[0]) + f(nodes[-1])
    for i in range(1, n):
        if i % 2 == 0:
            integral += 2 * f(nodes[i])
        else:
            integral += 4 * f(nodes[i])
    return (h / 3) * integral

def euler_rule(a, b, n):
    nodes, h = create_uniform_grid(a, b, n)
    integral = (f(nodes[0]) + f(nodes[-1])) / 2
    for i in range(1, n):
        integral += f(nodes[i])
    return h * integral + (h**2 / 12) * (df(nodes[0]) - df(nodes[-1]))

def runge_romberg(I_h, I_h2, p):
    return I_h2 + (I_h2 - I_h) / (2**p - 1)

def calculate_errors(a, b, n):
    h = (b - a) / n
    errors = {}
    
    max_d2f = find_max_derivative(d2f, a, b)
    errors['midpoint'] = (b - a) * h**2 / 24 * max_d2f
    
    errors['trapezoidal'] = (b - a) * h**2 / 12 * max_d2f
    
    max_d4f = find_max_derivative(d4f, a, b)
    errors['simpson'] = (b - a) * h**4 / 180 * max_d4f
    
    errors['euler'] = (b - a) * h**4 / 720 * max_d4f
    
    return errors

def main():
    a = -2
    b = 1.5
    epsilon = 1e-6  
    
    n = calculate_optimal_n(a, b, epsilon)
    
    print(f"Вычисление интеграла ∫[{a}, {b}] x/(x²+x+1) dx")
    print(f"Заданная точность: {epsilon}")
    print(f"Автоматически выбрано число отрезков: {n}")
    print("=" * 60)
    
    nodes_h, h = create_uniform_grid(a, b, n)
    nodes_h2, h2 = create_uniform_grid(a, b, 2*n)
    
    print(f"Равномерная сетка с шагом h: {len(nodes_h)} узлов")
    print(f"Равномерная сетка с шагом h/2: {len(nodes_h2)} узлов")
    
    print("Значения интегралов с шагом h:")
    I_mid_h = midpoint_rule(a, b, n)
    I_trap_h = trapezoidal_rule(a, b, n)
    I_simp_h = simpson_rule(a, b, n)
    I_euler_h = euler_rule(a, b, n)

    print(f"Средние прямоугольники: {I_mid_h:.8f}")
    print(f"Трапеции:               {I_trap_h:.8f}")
    print(f"Симпсон:                {I_simp_h:.8f}")
    print(f"Эйлер:                  {I_euler_h:.8f}")
    print()

    print(f"Значения интегралов с шагом h/2 (n = {2*n}):")
    I_mid_h2 = midpoint_rule(a, b, 2*n)
    I_trap_h2 = trapezoidal_rule(a, b, 2*n)
    I_simp_h2 = simpson_rule(a, b, 2*n)
    I_euler_h2 = euler_rule(a, b, 2*n)

    print(f"Средние прямоугольники: {I_mid_h2:.8f}")
    print(f"Трапеции:               {I_trap_h2:.8f}")
    print(f"Симпсон:                {I_simp_h2:.8f}")
    print(f"Эйлер:                  {I_euler_h2:.8f}")
    print()

    analytical_errors = calculate_errors(a, b, n)
    
    print("Аналитические оценки погрешностей:")
    print(f"Средние прямоугольники: {analytical_errors['midpoint']:.8f}")
    print(f"Трапеции:               {analytical_errors['trapezoidal']:.8f}")
    print(f"Симпсон:                {analytical_errors['simpson']:.8f}")
    print(f"Эйлер:                  {analytical_errors['euler']:.8f}")
    print()

    print("Уточненные значения по Рунге-Ромбергу:")
    p_mid = 2   
    p_trap = 2  
    p_simp = 4  
    p_euler = 4

    I_mid_rr = runge_romberg(I_mid_h, I_mid_h2, p_mid)
    I_trap_rr = runge_romberg(I_trap_h, I_trap_h2, p_trap)
    I_simp_rr = runge_romberg(I_simp_h, I_simp_h2, p_simp)
    I_euler_rr = runge_romberg(I_euler_h, I_euler_h2, p_euler)

    print(f"Средние прямоугольники: {I_mid_rr:.8f}")
    print(f"Трапеции:               {I_trap_rr:.8f}")
    print(f"Симпсон:                {I_simp_rr:.8f}")
    print(f"Эйлер:                  {I_euler_rr:.8f}")
    print()

    print("Практические оценки погрешностей (|I_h - I_h2|):")
    print(f"Средние прямоугольники: {abs(I_mid_h - I_mid_h2):.8f}")
    print(f"Трапеции:               {abs(I_trap_h - I_trap_h2):.8f}")
    print(f"Симпсон:                {abs(I_simp_h - I_simp_h2):.8f}")
    print(f"Эйлер:                  {abs(I_euler_h - I_euler_h2):.8f}")

if __name__ == "__main__":
    main()