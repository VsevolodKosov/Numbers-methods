import numpy as np

def thomas_algorithm(a, b, c, f):
    n = len(f)
    
    # Прогоночные коэффициенты
    alpha = np.zeros(n+1)  # α[1]...α[8]
    beta = np.zeros(n+1)   # β[1]...β[8]
    x = np.zeros(n)        # решение
    
    # Прямой ход
    # i=1 (первое уравнение)
    alpha[2] = -c[0] / b[0]
    beta[2] = f[0] / b[0]
    
    # i=2 до i=7
    for i in range(1, n-1):
        denominator = a[i] * alpha[i+1] + b[i]
        alpha[i+2] = -c[i] / denominator
        beta[i+2] = (f[i] - a[i] * beta[i+1]) / denominator
    
    # Обратный ход
    # i=8 (последнее уравнение)
    denominator = a[n-1] * alpha[n] + b[n-1]
    x[n-1] = (f[n-1] - a[n-1] * beta[n]) / denominator
    
    # i=7 down to i=1
    for i in range(n-2, -1, -1):
        x[i] = alpha[i+2] * x[i+1] + beta[i+2]
    
    return x

# Диагонали матрицы
a = np.array([0, 3, 2, -2, 3, -5, 3, 3])    # поддиагональ
b = np.array([7, 5, 9, 7, 8, 9, -6, 8])     # главная диагональ
c = np.array([-3, -2, -1, -3, 1, 4, -2, 0]) # наддиагональ
f = np.array([26, 28, 15, 7, -23, 24, -3, 24]) # правая часть

# Решение
solution = thomas_algorithm(a, b, c, f)
print("Решение системы:")
for i in range(len(solution)):
    print(f"x{i+1} = {solution[i]:.6f}")

# Проверка
print("\nПроверка:")
for i in range(8):
    # Вычисляем левую часть i-го уравнения
    left_side = 0
    if i == 0: left_side = b[0]*solution[0] + c[0]*solution[1]
    elif i == 7: left_side = a[7]*solution[6] + b[7]*solution[7]
    else: left_side = a[i]*solution[i-1] + b[i]*solution[i] + c[i]*solution[i+1]
    
    print(f"Уравнение {i+1}: {left_side:.2f} ≈ {f[i]} (погрешность: {abs(left_side - f[i]):.6f})")