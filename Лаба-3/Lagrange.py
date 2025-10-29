import matplotlib.pyplot as plt
import numpy as np

def lagrange_interpolation(x, x_points, y_points):
    """
    Вычисляет значение интерполяционного многочлена Лагранжа в точке x
    """
    n = len(x_points)
    result = 0.0
    
    for i in range(n):
        term = y_points[i]
        for j in range(n):
            if i != j:
                term *= (x - x_points[j]) / (x_points[i] - x_points[j])
        result += term
    
    return result

def get_points_by_numbers(numbers, x_data, y_data):
    """
    Выбирает точки по номерам (начиная с 1)
    """
    x_selected = []
    y_selected = []
    for num in numbers:
        idx = num - 1  # Преобразуем номер в индекс
        x_selected.append(x_data[idx])
        y_selected.append(y_data[idx])
    return x_selected, y_selected

def divided_difference(x, y):
    """
    Вычисляет разделенную разность для набора точек
    """
    n = len(x)
    if n == 0:
        return 0
    f = y.copy()
    for j in range(1, n):
        for i in range(n-1, j-1, -1):
            f[i] = (f[i] - f[i-1]) / (x[i] - x[i-j])
    return f[-1]

def calculate_error(x_target, selected_numbers, x_data, y_data, degree):
    """
    Вычисляет оценку погрешности интерполяции с использованием разделенных разностей
    """
    # Преобразуем selected_numbers в индексы (вычитаем 1) и получаем x_nodes, y_nodes
    selected_indices = [num-1 for num in selected_numbers]
    x_nodes = [x_data[i] for i in selected_indices]
    y_nodes = [y_data[i] for i in selected_indices]

    # Находим дополнительную точку
    all_indices = set(range(len(x_data)))
    remaining_indices = list(all_indices - set(selected_indices))
    
    # Если нет дополнительных точек, возвращаем 0
    if not remaining_indices:
        return 0.0
        
    # Сортируем оставшиеся индексы по расстоянию от x_target
    remaining_indices.sort(key=lambda i: abs(x_data[i] - x_target))

    additional_index = None
    min_x = min(x_nodes)
    max_x = max(x_nodes)
    
    # Ищем точку в интервале [min_x, max_x]
    for idx in remaining_indices:
        if min_x <= x_data[idx] <= max_x:
            additional_index = idx
            break

    if additional_index is None:
        additional_index = remaining_indices[0]  # ближайшая из оставшихся

    # Формируем список из (degree+2) точек: выбранные точки + дополнительная
    all_selected_indices = selected_indices + [additional_index]
    # Сортируем все точки по x
    all_selected_indices.sort(key=lambda i: x_data[i])
    x_all = [x_data[i] for i in all_selected_indices]
    y_all = [y_data[i] for i in all_selected_indices]

    # Вычисляем разделенную разность для этих точек
    dd = divided_difference(x_all, y_all)

    # Вычисляем произведение (x_target - x_i) для всех выбранных точек
    product = 1.0
    for x_node in x_nodes:
        product *= (x_target - x_node)

    error_estimate = abs(product * dd)
    return error_estimate

def create_plot(ax, x_data, y_data, x_target, points_2, points_3):
    """
    Создает график для заданной комбинации точек
    """
    # Получаем координаты выбранных точек
    x_nodes_2, y_nodes_2 = get_points_by_numbers(points_2, x_data, y_data)
    x_nodes_3, y_nodes_3 = get_points_by_numbers(points_3, x_data, y_data)
    
    # Вычисление интерполяции
    y_lagrange_2 = lagrange_interpolation(x_target, x_nodes_2, y_nodes_2)
    y_lagrange_3 = lagrange_interpolation(x_target, x_nodes_3, y_nodes_3)
    
    # Вычисление погрешностей
    error_2 = calculate_error(x_target, points_2, x_data, y_data, 2)
    error_3 = calculate_error(x_target, points_3, x_data, y_data, 3)
    
    # Вычисление разницы между интерполяциями
    diff = abs(y_lagrange_2 - y_lagrange_3)
    
    # Определяем диапазон для построения графиков на основе выбранных узлов
    all_nodes_x = list(set(x_nodes_2 + x_nodes_3))
    x_min = min(all_nodes_x) - 0.1
    x_max = max(all_nodes_x) + 0.1
    
    # Создаем массив точек для построения кривых интерполяции
    x_plot = np.linspace(x_min, x_max, 200)
    
    # Вычисляем значения интерполяционных многочленов для графика
    y_plot_2 = [lagrange_interpolation(x, x_nodes_2, y_nodes_2) for x in x_plot]
    y_plot_3 = [lagrange_interpolation(x, x_nodes_3, y_nodes_3) for x in x_plot]
    
    # Исходные данные (все точки черными кружками)
    ax.plot(x_data, y_data, 'ko', markersize=4, label='Исходные данные', zorder=3)
    
    # Узлы квадратичной интерполяции (красные)
    ax.plot(x_nodes_2, y_nodes_2, 'ro', markersize=8, label=f'L2 узлы: {points_2}', zorder=4)
    
    # Узлы кубической интерполяции (синие)
    ax.plot(x_nodes_3, y_nodes_3, 'bo', markersize=8, label=f'L3 узлы: {points_3}', zorder=4)
    
    # Целевая точка (зеленый X)
    ax.plot(x_target, y_lagrange_2, 'gx', markersize=10, markeredgewidth=2, 
            label=f'x={x_target}', zorder=6)
    
    # Линии интерполяционных многочленов
    ax.plot(x_plot, y_plot_2, 'r-', linewidth=2, alpha=0.8, zorder=2)
    ax.plot(x_plot, y_plot_3, 'b-', linewidth=2, alpha=0.8, zorder=2)
    
    # Настройка графика
    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('y', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Устанавливаем границы графика для показа всех данных
    ax.set_xlim(min(x_data) - 0.2, max(x_data) + 0.2)
    ax.set_ylim(min(y_data) - 0.5, max(y_data) + 0.5)
    
    # Добавляем аннотации для всех исходных точек с номерами
    for i, (x, y) in enumerate(zip(x_data, y_data)):
        ax.annotate(f'{i+1}', (x, y), 
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=6, alpha=0.6, color='gray')
    
    # Добавляем аннотации с результатами интерполяции
    ax.annotate(f'L2: {y_lagrange_2:.4f}', 
                (x_target, y_lagrange_2), 
                xytext=(15, 10), textcoords='offset points', 
                fontsize=8, alpha=0.9, color='red',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))
    
    ax.annotate(f'L3: {y_lagrange_3:.4f}', 
                (x_target, y_lagrange_3), 
                xytext=(15, -25), textcoords='offset points', 
                fontsize=8, alpha=0.9, color='blue',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
                arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7))
    
    # Создаем кастомную легенду с значениями L2, L3, погрешностями и разницей
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=6, label='Исходные данные'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=6, label=f'L2 узлы: {points_2}'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=6, label=f'L3 узлы: {points_3}'),
        Line2D([0], [0], marker='x', color='green', markeredgewidth=2, markersize=8, label=f'x={x_target}'),
        Line2D([0], [0], color='red', linewidth=2, label=f'L2 = {y_lagrange_2:.4f}'),
        Line2D([0], [0], color='blue', linewidth=2, label=f'L3 = {y_lagrange_3:.4f}'),
        Line2D([0], [0], color='w', linewidth=0, label=f'R2 = {error_2:.4f}'),
        Line2D([0], [0], color='w', linewidth=0, label=f'R3 = {error_3:.4f}'),
        Line2D([0], [0], color='w', linewidth=0, label=f'|L2-L3| = {diff:.4f}')
    ]
    
    ax.legend(handles=legend_elements, fontsize=7, loc='best')

def main():
    """
    Основная функция программы
    """
    x_data = [-0.55, -0.14, 0.27, 0.68, 1.09, 1.50, 1.91, 2.32, 2.73]
    y_data = [2.374, 4.213, 4.986, 4.132, 4.128, 2.615, 1.877, 1.684, 0.219]
    x_target = 1.721

    print("=" * 60)
    print("ИНТЕРПОЛЯЦИЯ ТАБЛИЧНО ЗАДАННОЙ ФУНКЦИИ")
    print("=" * 60)
    
    print("\nИсходные данные:")
    print("№\tx\t\ty")
    for i in range(len(x_data)):
        print(f"{i+1}\t{x_data[i]:.2f}\t\t{y_data[i]:.3f}")
    
    print(f"\nВычисление в точке x = {x_target}")
    
    # Захардкоженные комбинации точек
    combinations = [
        ([5, 6, 7], [6, 7, 8, 9]),
        ([5, 6, 7], [5, 6, 7, 8]),
        ([5, 6, 7], [4, 5, 6, 7]),
        ([6, 7, 8], [6, 7, 8, 9]),
        ([6, 7, 8], [5, 6, 7, 8]),
        ([6, 7, 8], [4, 5, 6, 7])
    ]
    
    # Создаем фигуру с 6 подграфиками (2 строки, 3 столбца)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Строим графики для каждой комбинации
    for i, (points_2, points_3) in enumerate(combinations):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        # Получаем координаты выбранных точек
        x_nodes_2, y_nodes_2 = get_points_by_numbers(points_2, x_data, y_data)
        x_nodes_3, y_nodes_3 = get_points_by_numbers(points_3, x_data, y_data)
        
        # Вычисление интерполяции
        y_lagrange_2 = lagrange_interpolation(x_target, x_nodes_2, y_nodes_2)
        y_lagrange_3 = lagrange_interpolation(x_target, x_nodes_3, y_nodes_3)
        
        # Вычисление погрешностей
        error_2 = calculate_error(x_target, points_2, x_data, y_data, 2)
        error_3 = calculate_error(x_target, points_3, x_data, y_data, 3)
        
        # Вычисление разницы между интерполяциями
        diff = abs(y_lagrange_2 - y_lagrange_3)
        
        # Вывод результатов в консоль
        print(f"\n{'='*50}")
        print(f"Комбинация {i+1}: L2{points_2} и L3{points_3}")
        print(f"L2({x_target}) = {y_lagrange_2:.6f}, |R₂| = {error_2:.6f}")
        print(f"L3({x_target}) = {y_lagrange_3:.6f}, |R₃| = {error_3:.6f}")
        print(f"|L2 - L3| = {diff:.6f}")
        
        # Создаем график
        create_plot(ax, x_data, y_data, x_target, points_2, points_3)
    
    plt.tight_layout()
    plt.show()

# Точка входа
if __name__ == "__main__":
    main()