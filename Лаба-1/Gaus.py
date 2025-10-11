def gaussian_matrix_operations(A, b):
    n = len(A)

    # Создаем расширенную матрицу [A | I | b]
    augmented = []
    for i in range(n):
        row = A[i][:] + [1 if j == i else 0 for j in range(n)] + [b[i]]
        augmented.append(row)
    cols = 2 * n + 1

    swaps = 0
    det_factor = 1.0

    # Прямой ход метода Гаусса
    for col in range(n):
        # Поиск строки с максимальным элементом в текущем столбце
        max_row = col
        for i in range(col + 1, n):
            if abs(augmented[i][col]) > abs(augmented[max_row][col]):
                max_row = i

        # Перестановка строк при необходимости
        if max_row != col:
            augmented[col], augmented[max_row] = augmented[max_row], augmented[col]
            swaps += 1
            det_factor *= -1

        pivot = augmented[col][col]

        # Проверка на нулевой главный элемент
        if abs(pivot) < 1e-12:
            det = 0.0
            inv = None
            sol = None
            return det, inv, sol

        det_factor *= pivot

        # Нормализация ведущей строки
        for j in range(col, cols):
            augmented[col][j] /= pivot

        # Обнуление остальных элементов в столбце
        for i in range(n):
            if i == col:
                continue
            factor = augmented[i][col]
            for j in range(col, cols):
                augmented[i][j] -= factor * augmented[col][j]

    # Определитель
    det = det_factor

    # Извлечение обратной матрицы
    inv = []
    for i in range(n):
        inv.append(augmented[i][n:2 * n])

    # Извлечение решения
    sol = []
    for i in range(n):
        sol.append(augmented[i][2 * n])

    return det, inv, sol


if __name__ == "__main__":
    A = [
        [3, 6, -4, 3, 2],
        [4, 2, 1, 3, 5],
        [-2, 3, 3, 2, 9],
        [2, -5, -4, 0, 3],
        [9, -4, 5, 1, -2]
    ]
    b = [15, 58, 72, 39, 24]

    det, inv, sol = gaussian_matrix_operations(A, b)

    print("Результаты метода Гаусса:")
    print("\nРешение системы:")
    print([round(x, 4) for x in sol])

    print(f"\nОпределитель: {round(det, 4)}")

    print("\nОбратная матрица:")
    for row in inv:
        print([round(x, 4) for x in row])
