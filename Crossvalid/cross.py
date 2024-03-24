from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

data = load_breast_cancer()
X = data.data
y = data.target

knn = KNeighborsClassifier(n_neighbors=7)

scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')

print("Точность для каждого фолда:")
for i, score in enumerate(scores):
    print(f"Фолд {i+1}: {score}")

print("Средняя точность: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
