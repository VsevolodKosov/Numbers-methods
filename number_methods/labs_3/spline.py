import sys
import os
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from labs_1.progon import progon

def cubic_spline_natural(x, y, point):
    n = len(x) - 1
    
    h = [x[i] - x[i-1] for i in range(1, n+1)]
    
    v = []
    for i in range(2, n):
        term1 = (y[i] - y[i-1]) / h[i-1]
        term2 = (y[i-1] - y[i-2]) / h[i-2]
        v.append(3 * (term1 - term2))
    
    A = [[0.0] * (n-2) for _ in range(n-2)]
    
    for i in range(n-2):
        A[i][i] = 2 * (h[i] + h[i+1])
        if i < n-3:
            A[i][i+1] = h[i+1]
        if i > 0:
            A[i][i-1] = h[i]
    
    c_inner, stability, check = progon(A, v)
    
    c = [0.0] * (n + 2)
    c[1] = 0.0
    for i in range(2, n):
        c[i] = c_inner[i-2]
    c[n] = 0.0

    a_coeff = [0.0] * (n + 1)
    b_coeff = [0.0] * (n + 1)
    d_coeff = [0.0] * (n + 1)
    
    for i in range(1, n + 1):
        a_coeff[i] = y[i-1]
    
    for i in range(1, n):
        b_coeff[i] = (y[i] - y[i-1]) / h[i-1] - (h[i-1] / 3) * (c[i+1] + 2 * c[i])
    
    b_coeff[n] = (y[n] - y[n-1]) / h[n-1] - (2/3) * h[n-1] * c[n]
    
    for i in range(1, n):
        d_coeff[i] = (c[i+1] - c[i]) / (3 * h[i-1])
    
    d_coeff[n] = -c[n] / (3 * h[n-1])
    
    segment_index = None
    for i in range(1, n + 1):
        if x[i-1] <= point <= x[i]:
            segment_index = i
            break
    
    delta_x = point - x[segment_index-1]

    spline_value = (a_coeff[segment_index] + b_coeff[segment_index] * delta_x + c[segment_index] * delta_x**2 + d_coeff[segment_index] * delta_x**3)
    
    spline_derivative = (b_coeff[segment_index] + 2 * c[segment_index] * delta_x + 3 * d_coeff[segment_index] * delta_x**2)
    
    spline_second_derivative = (2 * c[segment_index] + 6 * d_coeff[segment_index] * delta_x)
    
    return spline_value, spline_derivative, spline_second_derivative, a_coeff, b_coeff, c, d_coeff, segment_index

def plot_spline(x, y, point, a_coeff, b_coeff, c, d_coeff, segment_index):
    fig = plt.figure(figsize=(20, 16))
    
    n = len(x) - 1
    
    S_point, S_der_point, S_der2_point, _, _, _, _, _ = cubic_spline_natural(x, y, point)
    
    ax1 = plt.subplot(2, 3, 1)
    ax2 = plt.subplot(2, 3, 2)
    ax3 = plt.subplot(2, 3, 3)
    ax4 = plt.subplot(2, 3, 4)
    ax5 = plt.subplot(2, 3, 5)
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    ax1.scatter(x, y, color='red', s=50, zorder=5, label='Узловые точки')
    ax1.scatter([point], [S_point], color='green', s=150, zorder=6, marker='*', label=f'x* = {point}')
    ax1.axvline(x=point, color='green', linestyle='--', alpha=0.7, label=f'S(x*) = {S_point:.4f}')
    
    for i in range(1, n + 1):
        x_segment = np.linspace(x[i-1], x[i], 100)
        dx_segment = x_segment - x[i-1]
        y_segment = a_coeff[i] + b_coeff[i] * dx_segment + c[i] * dx_segment**2 + d_coeff[i] * dx_segment**3
        ax1.plot(x_segment, y_segment, 'b-', linewidth=2)
    
    ax1.set_title('Кубический сплайн и исходные данные', pad=15)
    ax1.set_xlabel('x')
    ax1.set_ylabel('S(x)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    for i in range(1, n + 1):
        x_segment = np.linspace(x[i-1], x[i], 100)
        dx_segment = x_segment - x[i-1]
        y_segment = a_coeff[i] + b_coeff[i] * dx_segment + c[i] * dx_segment**2 + d_coeff[i] * dx_segment**3
        color = colors[(i-1) % len(colors)]
        ax2.plot(x_segment, y_segment, color=color, linewidth=2, label=f'Сегмент {i}')
    
    ax2.scatter(x, y, color='red', s=50, zorder=5, label='Узловые точки')
    ax2.set_title('Сегменты кубического сплайна', pad=15)
    ax2.set_xlabel('x')
    ax2.set_ylabel('S(x)')
    ax2.legend(fontsize='small', bbox_to_anchor=(-0.15, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    ax3.scatter([point], [S_der_point], color='green', s=150, zorder=6, marker='*', label=f'x* = {point}')
    ax3.axvline(x=point, color='green', linestyle='--', alpha=0.7, label=f'S\'(x*) = {S_der_point:.4f}')
    
    for i in range(1, n + 1):
        x_segment = np.linspace(x[i-1], x[i], 100)
        dx_segment = x_segment - x[i-1]
        y_derivative = b_coeff[i] + 2 * c[i] * dx_segment + 3 * d_coeff[i] * dx_segment**2
        ax3.plot(x_segment, y_derivative, 'g-', linewidth=2)
    
    for i in range(len(x)):
        if i == 0:
            der = b_coeff[1]
        elif i == n:
            der = b_coeff[n] + 2 * c[n] * (x[n] - x[n-1]) + 3 * d_coeff[n] * (x[n] - x[n-1])**2
        else:
            der = b_coeff[i] + 2 * c[i] * (x[i] - x[i-1]) + 3 * d_coeff[i] * (x[i] - x[i-1])**2
        ax3.scatter(x[i], der, color='red', s=30, zorder=5)
    
    ax3.set_title('Первая производная кубического сплайна S\'(x)', pad=15)
    ax3.set_xlabel('x')
    ax3.set_ylabel('S\'(x)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4.scatter([point], [S_der2_point], color='green', s=150, zorder=6, marker='*', label=f'x* = {point}')
    ax4.axvline(x=point, color='green', linestyle='--', alpha=0.7, label=f'S\'\'(x*) = {S_der2_point:.4f}')
    
    for i in range(1, n + 1):
        x_segment = np.linspace(x[i-1], x[i], 100)
        dx_segment = x_segment - x[i-1]
        y_second_derivative = 2 * c[i] + 6 * d_coeff[i] * dx_segment
        ax4.plot(x_segment, y_second_derivative, 'r-', linewidth=2)
    
    for i in range(len(x)):
        if i == 0:
            der2 = 2 * c[1]
        elif i == n:
            der2 = 2 * c[n] + 6 * d_coeff[n] * (x[n] - x[n-1])
        else:
            der2 = 2 * c[i] + 6 * d_coeff[i] * (x[i] - x[i-1])
        ax4.scatter(x[i], der2, color='red', s=30, zorder=5)
    
    ax4.set_title('Вторая производная кубического сплайна S\'\'(x)', pad=15)
    ax4.set_xlabel('x')
    ax4.set_ylabel('S\'\'(x)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    i = segment_index
    x_segment_detailed = np.linspace(x[i-1], x[i], 200)
    dx_segment_detailed = x_segment_detailed - x[i-1]
    y_segment_detailed = a_coeff[i] + b_coeff[i] * dx_segment_detailed + c[i] * dx_segment_detailed**2 + d_coeff[i] * dx_segment_detailed**3
    
    ax5.plot(x_segment_detailed, y_segment_detailed, 'b-', linewidth=3, label=f'Кубический сплайн (сегмент {i})')
    
    for j in range(len(x)):
        ax5.scatter(x[j], y[j], color='red', s=50, zorder=5, label='Узловые точки' if j == 0 else "")
    
    ax5.scatter([point], [S_point], color='green', s=150, zorder=6, marker='*', label=f'x* = {point}')
    ax5.axvline(x=point, color='green', linestyle='--', alpha=0.7, label=f'S(x*) = {S_point:.4f}')
    
    margin = (max(x) - min(x)) * 0.1 
    ax5.set_xlim(min(x) - margin, max(x) + margin)
    ax5.set_ylim(min(y) - 0.5, max(y) + 0.5)
    
    ax5.set_title(f'Детальный вид вокруг x* (сегмент {segment_index})', pad=15)
    ax5.set_xlabel('x')
    ax5.set_ylabel('S(x)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(top=0.92, hspace=0.4, wspace=0.3)
    plt.show()

def main():
    x = [-2.0, -1.72, -1.36, -0.96, -0.64, -0.28, 0.04, 0.36, 0.80, 1.56, 2.0]
    y = [-3.254, -4.524, -3.203, 0.231, -0.123, 1.549, 1.961, 1.028, 1.126, -1.328, -1.904]
    point = 0.653
    
    S_3, S_3_der, S_3_der2, a_coeff, b_coeff, c, d_coeff, segment_index = cubic_spline_natural(x, y, point)
    
    print("Результаты расчета кубического сплайна:")
    print(f"S_3(x*) = {S_3:.6f}")
    print(f"S'_3(x*) = {S_3_der:.6f}")
    print(f"S''_3(x*) = {S_3_der2:.6f}")
    print(f"Сегмент: {segment_index}")
    
    plot_spline(x, y, point, a_coeff, b_coeff, c, d_coeff, segment_index)

if __name__ == "__main__":
    main()