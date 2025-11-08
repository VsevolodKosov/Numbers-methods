import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from labs_1.Gaus import method_Gaus

# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

def create_vandermonde_matrix_mnk(x, degree):
    return [[xi**j for j in range(0, degree + 1)] for xi in x]

def multiply_matrices_mnk(A, B):
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    
    if cols_A != rows_B:
        raise ValueError("Несовместимые размеры матриц")
    
    result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]
    
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]
    
    return result

def transpose_matrix_mnk(A):
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]

def evaluate_polynomial(coeffs, x_point):
    result = 0
    for power, coeff in enumerate(coeffs):
        result += coeff * (x_point ** power)
    return result

def calculate_errors_mnk(y_true, y_pred):
    """Расчет ошибок строго по методичке стр. 5"""
    n_points = len(y_true)  # количество точек
    sum_squared_errors = sum((y_pred_i - y_true_i)**2 for y_pred_i, y_true_i in zip(y_pred, y_true))
    standard_error = (sum_squared_errors / n_points)**0.5  # делим на количество точек
    return sum_squared_errors, standard_error

# ==================== ОСНОВНАЯ ФУНКЦИЯ МНК ====================

def calculate_approximating_polynomials(x, y, degrees, x_star=None):
    polynomials = {}
    
    for degree in degrees:
        V = create_vandermonde_matrix_mnk(x, degree)
        
        V_transposed = transpose_matrix_mnk(V)
        A_matrix = multiply_matrices_mnk(V_transposed, V)
        
        b_vector = [sum(V_transposed[i][j] * y[j] for j in range(len(y))) for i in range(len(V_transposed))]
        
        det, inv, coefficients = method_Gaus(A_matrix, b_vector)
        
        poly_values = [evaluate_polynomial(coefficients, xi) for xi in x]
        
        sum_squared_errors, standard_error = calculate_errors_mnk(y, poly_values)
        
        poly_value_at_star = evaluate_polynomial(coefficients, x_star) if x_star is not None else None
        
        polynomials[degree] = {
            'coefficients': coefficients,
            'values': poly_values,
            'sum_squared_errors': sum_squared_errors,
            'standard_error': standard_error,
            'value_at_star': poly_value_at_star,
            'function_name': f'F{degree}(x)'
        }
    
    return polynomials

# ==================== ФУНКЦИИ ВЫВОДА И ВИЗУАЛИЗАЦИИ ====================

def print_results(polynomials, x_star):
    print("=" * 70)
    print("МЕТОД НАИМЕНЬШИХ КВАДРАТОВ - ПРИБЛИЖАЮЩИЕ МНОГОЧЛЕНЫ")
    print("=" * 70)
    
    print(f'x* = {x_star}')
    for degree, poly_data in polynomials.items():
        coeffs = poly_data['coefficients']
        
        # Прямой порядок коэффициентов как в методичке
        poly_terms = []
        for power, coeff in enumerate(coeffs):
            if power == 0:
                poly_terms.append(f"{coeff:.4f}")
            elif power == 1:
                poly_terms.append(f"{coeff:.4f}x")
            else:
                poly_terms.append(f"{coeff:.4f}x^{power}")
        poly_function_str = " + ".join(poly_terms)
        
        print(f"\nПРИБЛИЖАЮЩИЙ МНОГОЧЛЕН {degree} СТЕПЕНИ:")
        print(f"F{degree}(x) = {poly_function_str}")
        print(f"F{degree}(x*): {poly_data['value_at_star']:.4f}")
        print(f"Сумма квадратов ошибок: Φ_{degree} = {poly_data['sum_squared_errors']:.4f}")
        print(f"Среднеквадратичное отклонение: E_{degree} = {poly_data['standard_error']:.4f}")

def plot_polynomials_and_function(x, y, polynomials, x_star):
    plt.figure(figsize=(12, 8))
    
    plt.scatter(x, y, color='black', s=80, label='Приближаемая функция (исходные данные)', zorder=5)
    
    colors = ['red', 'green', 'blue']
    markers = ['o', 's', '^']
    
    for i, (degree, poly_data) in enumerate(polynomials.items()):
        x_plot = [i * 0.01 for i in range(0, 201)]  
        coeffs = poly_data['coefficients']
        y_plot = [evaluate_polynomial(coeffs, xi) for xi in x_plot]
        plt.plot(x_plot, y_plot, color=colors[i], linewidth=2, 
                label=f'Приближающий многочлен {degree} степени F{degree}(x)')
        
        plt.scatter(x, poly_data['values'], color=colors[i], s=40, 
                   marker=markers[i], alpha=0.7, 
                   label=f'F{degree}(x) в точках данных')
    
    star_y_values = [poly_data['value_at_star'] for poly_data in polynomials.values()]
    plt.scatter([x_star] * len(star_y_values), star_y_values, 
               color='purple', s=100, marker='*', 
               label=f'Значения в точке x* = {x_star}', zorder=6)
    
    for i, (degree, poly_data) in enumerate(polynomials.items()):
        if i % 2 == 0:
            plt.annotate(f'F{degree}({x_star}) = {poly_data["value_at_star"]:.3f}', 
                        (x_star, poly_data['value_at_star']),
                        xytext=(-120, 10+i*20), textcoords='offset points',
                        fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                        arrowprops=dict(arrowstyle="->", color='gray', alpha=0.7))
        else:
            plt.annotate(f'F{degree}({x_star}) = {poly_data["value_at_star"]:.3f}', 
                        (x_star, poly_data['value_at_star']),
                        xytext=(20, 10+(i-1)*20), textcoords='offset points',
                        fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                        arrowprops=dict(arrowstyle="->", color='gray', alpha=0.7))
    
    plt.axvline(x=x_star, color='gray', linestyle='--', alpha=0.5)
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Приближающие многочлены и приближаемая функция')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ==================== Точка входа ====================
def main():
    x = [0.00, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    y = [3.4927, 3.1831, 1.8074, 2.0819, 1.3693, 1.5125, 0.6283, 0.8513, 1.7641, 3.7148, 4.1283]
    x_star = 1.295
    
    degrees = [1, 2, 3]
    polynomials = calculate_approximating_polynomials(x, y, degrees, x_star)
    
    print_results(polynomials, x_star)
    
    plot_polynomials_and_function(x, y, polynomials, x_star)
    
if __name__ == "__main__":
    main()