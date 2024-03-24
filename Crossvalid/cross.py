import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor

# Загрузка набора данных о пациентах с диабетом
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Определение диапазона значений гиперпараметра k
k_values = list(range(1, 21))

# Список для сохранения средних оценок точности для каждого значения k
mean_accuracies = []

# Перебор всех значений k
for k in k_values:
    # Создание регрессора kNN
    knn = KNeighborsRegressor(n_neighbors=k)
    # Выполнение кросс-валидации и вычисление средней оценки точности
    scores = cross_val_score(knn, X, y, cv=5, scoring='neg_mean_squared_error')
    mean_accuracy = -np.mean(scores)  # обратим результат, так как scoring='neg_mean_squared_error'
    mean_accuracies.append(mean_accuracy)
    print(f"k = {k}, Среднее квадратичное отклонение = {mean_accuracy:.2f}")

# Вывод наилучшей оценки и соответствующего значения k
best_accuracy = min(mean_accuracies)
best_k = k_values[mean_accuracies.index(best_accuracy)]
print(f"\nНаилучшее среднее квадратичное отклонение: {best_accuracy:.2f}, достигается при k = {best_k}.")
