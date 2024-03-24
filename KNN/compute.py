from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_alt, X_test_alt, y_train_alt, y_test_alt = train_test_split(X, y, test_size=0.2, random_state=1)