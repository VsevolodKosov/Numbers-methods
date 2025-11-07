import math
import matplotlib.pyplot as plt
import numpy as np

def f(X):
    x1, x2 = X
    return [x1**2 - math.exp(x2) + 1,
            x2**2 - 2*math.sin(x1) - 3]

def J(X):
    x1, x2 = X
    return [[2*x1, -math.exp(x2)],
            [-2*math.cos(x1), 2*x2]]

def solve_linear_2x2(A, b):
    a11, a12 = A[0]
    a21, a22 = A[1]
    b1, b2 = b
    det = a11*a22 - a12*a21
    if det == 0:
        raise ValueError("Матрица вырождена")
    x = (b1*a22 - b2*a12) / det
    y = (a11*b2 - a21*b1) / det
    return [x, y]

def phi1(x2):
    return math.sqrt(math.exp(x2) - 1)

def phi2(x1):
    return math.sqrt(3 + 2*math.sin(x1))

def J_Phi(X):
    x1, x2 = X
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

def matrix_determinant(J_matrix):
    a11, a12 = J_matrix[0]
    a21, a22 = J_matrix[1]
    return a11 * a22 - a12 * a21

def check_convergence_condition(X, lambda_val):
    J_phi = J_Phi(X)
    
    M = [
        [1 - lambda_val + lambda_val * J_phi[0][0], lambda_val * J_phi[0][1]],
        [lambda_val * J_phi[1][0], 1 - lambda_val + lambda_val * J_phi[1][1]]
    ]
    
    det_M = matrix_determinant(M)
    return abs(det_M) < 1

def newton_method_v1(F, JF, X0, eps):
    X = X0[:]
    iterations = 0
    while True:
        try:
            F_X = F(X)
            J_X = JF(X)
            
            J_inv_f = solve_linear_2x2(J_X, F_X)
            
            X_new = [X[j] - J_inv_f[j] for j in range(len(X))]
            iterations += 1
            
            max_change = max([abs(X_new[j] - X[j]) for j in range(len(X))])
            X = X_new
            
            if max_change < eps:
                return X, iterations, True
        except (ValueError, ZeroDivisionError, OverflowError) as e:
            return X, iterations, False

def newton_method_v2(F, JF, X0, eps):
    X = X0[:]
    iterations = 0
    while True:
        try:
            F_X = F(X)
            J_X = JF(X)
            
            a11, a12 = J_X[0]
            a21, a22 = J_X[1]
            
            f1, f2 = F_X
            
            det_J = a11 * a22 - a12 * a21
            
            if det_J == 0:
                return X, iterations, False
            
            det_A1 = f1 * a22 - f2 * a12
            
            det_A2 = a11 * f2 - a21 * f1
            
            delta_x1 = -det_A1 / det_J
            delta_x2 = -det_A2 / det_J
            
            X_new = [X[0] + delta_x1, X[1] + delta_x2]
            iterations += 1
            
            max_change = max([abs(X_new[j] - X[j]) for j in range(len(X))])
            X = X_new
            
            if max_change < eps:
                return X, iterations, True
        except (ValueError, ZeroDivisionError, OverflowError) as e:
            return X, iterations, False

def simple_iteration_fixed(X0, eps, lambda_val=0.4):
    X = X0[:]
    iterations = 0
    
    while True:
        try:
            x1_new = phi1(X[1])  
            x2_new = phi2(X[0])

            x1_relaxed = (1 - lambda_val) * X[0] + lambda_val * x1_new
            x2_relaxed = (1 - lambda_val) * X[1] + lambda_val * x2_new
            
            X_new = [x1_relaxed, x2_relaxed]
            iterations += 1
            
            max_change = max([abs(X_new[j] - X[j]) for j in range(len(X))])
            X = X_new
            
            if max_change < eps:
                return X, iterations, True, lambda_val
        except (ValueError, ZeroDivisionError, OverflowError) as e:
            return X, iterations, False, lambda_val

def seidel_method_fixed(X0, eps, lambda_val=0.4):
    X = X0[:]
    iterations = 0
    
    while True:
        try:
            X_old = X[:]
        
            x1_new = phi1(X[1])  
            X[0] = (1 - lambda_val) * X[0] + lambda_val * x1_new
            
            x2_new = phi2(X[0])  
            X[1] = (1 - lambda_val) * X[1] + lambda_val * x2_new
            
            iterations += 1
            
            max_change = max([abs(X[j] - X_old[j]) for j in range(len(X))])
            
            if max_change < eps:
                return X, iterations, True, lambda_val
        except (ValueError, ZeroDivisionError, OverflowError) as e:
            return X, iterations, False, lambda_val

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

    x1_from_x2 = []
    for x2 in x2_vals:
        val = math.exp(x2) - 1
        if val >= 0:
            sqrt_val = math.sqrt(val)
            x1_from_x2.append([sqrt_val, x2])
            x1_from_x2.append([-sqrt_val, x2])

    x2_from_x1 = []
    for x1 in x1_vals:
        val = 2*math.sin(x1) + 3
        if val >= 0:
            sqrt_val = math.sqrt(val)
            x2_from_x1.append([x1, sqrt_val])
            x2_from_x1.append([x1, -sqrt_val])

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

def plot_system(initial_approximations):
    plt.figure(figsize=(12, 10))

    x1 = np.linspace(-3, 3, 500)
    x2 = np.linspace(-3, 3, 500)
    X1, X2 = np.meshgrid(x1, x2)

    F1 = X1**2 - np.exp(X2) + 1
    F2 = X2**2 - 2*np.sin(X1) - 3

    contour1 = plt.contour(X1, X2, F1, levels=[0], colors='blue', linewidths=2)
    contour2 = plt.contour(X1, X2, F2, levels=[0], colors='red', linewidths=2)

    plt.clabel(contour1, inline=True, fontsize=10, fmt='F1=0')
    plt.clabel(contour2, inline=True, fontsize=10, fmt='F2=0')

    if initial_approximations:
        for i, approx in enumerate(initial_approximations):
            plt.plot(approx[0], approx[1], 'go', markersize=10, markeredgecolor='black', 
                    label='Нач. приближение' if i == 0 else "")
            plt.annotate(f'({approx[0]:.3f}, {approx[1]:.3f})', 
                        (approx[0], approx[1]), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

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

if __name__ == "__main__":
    eps = 1e-4
    lambda_val = 0.4

    selected_approximations = [
        [-1.332, 1.020],
        [2.557, 2.020],
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
    
    print(f"\nРезультаты сходимости для различных начальных приближений (λ = {lambda_val}):")
    
    for i, X0 in enumerate(selected_approximations):
        print(f"\n--- Начальное приближение {i+1}: {X0} ---")
        
        X_newton_v1, iter_newton_v1, conv_newton_v1 = newton_method_v1(f, J, X0, eps)
        X_newton_v2, iter_newton_v2, conv_newton_v2 = newton_method_v2(f, J, X0, eps)
        X_simple, iter_simple, conv_simple, lambda_used = simple_iteration_fixed(X0, eps, lambda_val)
        X_seidel, iter_seidel, conv_seidel, lambda_used = seidel_method_fixed(X0, eps, lambda_val)
        
        if conv_newton_v1:
            print(f"Метод Ньютона (способ 1): x = {X_newton_v1}, итерации = {iter_newton_v1}")
            roots_info.append({'root': X_newton_v1, 'method': 'Ньютон (1)', 'color': colors[0]})
        else:
            print("Метод Ньютона (способ 1): не сошелся")
            
        if conv_newton_v2:
            print(f"Метод Ньютона (способ 2): x = {X_newton_v2}, итерации = {iter_newton_v2}")
            roots_info.append({'root': X_newton_v2, 'method': 'Ньютон (2)', 'color': colors[1]})
        else:
            print("Метод Ньютона (способ 2): не сошелся")
            
        if conv_simple:
            print(f"Метод простых итераций: x = {X_simple}, итерации = {iter_simple}")
            roots_info.append({'root': X_simple, 'method': 'Простые итерации', 'color': colors[2]})
        else:
            print("Метод простых итераций: не сошелся")
            
        if conv_seidel:
            print(f"Метод Зейделя: x = {X_seidel}, итерации = {iter_seidel}")
            roots_info.append({'root': X_seidel, 'method': 'Зейделя', 'color': colors[3]})
        else:
            print("Метод Зейделя: не сошелся")