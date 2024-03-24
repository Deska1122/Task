import numpy as np


def euclidean_distance(point1, point2):
    return np.sqrt(np.sum(np.square(np.array(point1) - np.array(point2))))


def manhattan_distance(point1, point2):
    return np.sum(np.abs(np.array(point1) - np.array(point2)))


def chebyshev_distance(point1, point2):
    return np.max(np.abs(np.array(point1) - np.array(point2)))


point1 = [2, 3, 4]
point2 = [5, 6, 7]

print("Евклидово расстояние:", euclidean_distance(point1, point2))
print("Манхэттенское расстояние:", manhattan_distance(point1, point2))
print("Расстояние Чебышёва:", chebyshev_distance(point1, point2))
