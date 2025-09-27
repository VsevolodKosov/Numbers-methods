def gaussian_elimination(A, b):
    """
    Метод Гаусса для решения системы линейных уравнений Ax = b
    """
    n = len(A)

    # Преобразуем в расширенную матрицу
    for i in range(n):
        A[i] = A[i] + [b[i]]

    # Прямой ход
    for col in range(n):
        # Поиск ведущего элемента
        max_row = col
        for i in range(col + 1, n):
            if abs(A[i][col]) > abs(A[max_row][col]):
                max_row = i
        # Перестановка строк
        if max_row != col:
            A[col], A[max_row] = A[max_row], A[col]

        # Нормализация ведущей строки
        pivot = A[col][col]
        if abs(pivot) < 1e-12:
            raise ValueError("Система не имеет единственного решения")
        for j in range(col, n + 1):
            A[col][j] /= pivot

        # Обнуление остальных строк
        for i in range(n):
            if i != col:
                factor = A[i][col]
                for j in range(col, n + 1):
                    A[i][j] -= factor * A[col][j]

    # Решение
    x = [A[i][n] for i in range(n)]
    return x


def determinant(A):
    """
    Вычисление определителя методом Гаусса
    """
    n = len(A)
    M = [row[:] for row in A]  # копия матрицы
    det = 1
    swap_count = 0

    for col in range(n):
        # Поиск ведущего элемента
        max_row = col
        for i in range(col + 1, n):
            if abs(M[i][col]) > abs(M[max_row][col]):
                max_row = i
        if abs(M[max_row][col]) < 1e-12:
            return 0  # вырожденная матрица

        # Перестановка строк
        if max_row != col:
            M[col], M[max_row] = M[max_row], M[col]
            swap_count += 1

        # Диагональный элемент
        pivot = M[col][col]
        det *= pivot

        # Нормализация строки
        for j in range(col + 1, n):
            M[col][j] /= pivot

        # Обнуление под диагональю
        for i in range(col + 1, n):
            factor = M[i][col]
            for j in range(col + 1, n):
                M[i][j] -= factor * M[col][j]

    # Учитываем перестановки строк
    if swap_count % 2 == 1:
        det = -det

    return det


def inverse_matrix(A):
    """
    Метод Гаусса-Жордана для нахождения обратной матрицы
    """
    n = len(A)
    # Формируем расширенную матрицу [A | I]
    augmented = []
    for i in range(n):
        row = A[i][:] + [1 if j == i else 0 for j in range(n)]
        augmented.append(row)

    # Прямой и обратный ход
    for col in range(n):
        # Поиск ведущего элемента
        max_row = col
        for i in range(col + 1, n):
            if abs(augmented[i][col]) > abs(augmented[max_row][col]):
                max_row = i
        if max_row != col:
            augmented[col], augmented[max_row] = augmented[max_row], augmented[col]

        # Нормализация ведущей строки
        pivot = augmented[col][col]
        if abs(pivot) < 1e-12:
            raise ValueError("Матрица вырожденная, обратной нет")
        for j in range(2 * n):
            augmented[col][j] /= pivot

        # Обнуление остальных строк
        for i in range(n):
            if i != col:
                factor = augmented[i][col]
                for j in range(2 * n):
                    augmented[i][j] -= factor * augmented[col][j]

    # Извлекаем обратную матрицу
    inv = []
    for i in range(n):
        inv.append(augmented[i][n:])
    return inv


if __name__ == "__main__":
    # Пример системы
    A = [
        [3, 6, -4, 3, 2],
        [4, 2, 1, 3, 5],
        [-2, 3, 3, 2, 9],
        [2, -5, -4, 0, 3],
        [9, -4, 5, 1, -2]
    ]
    b = [15, 58, 72, 39, 24]

    print("Решение системы:")
    x = gaussian_elimination([row[:] for row in A], b[:])
    print(x)

    print("\nОпределитель:")
    print(determinant([row[:] for row in A]))

    print("\nОбратная матрица:")
    inv = inverse_matrix([row[:] for row in A])
    for row in inv:
        print(row)
