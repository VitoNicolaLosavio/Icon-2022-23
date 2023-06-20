from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


class KNN:
    def __init__(self, x_train, x_test, y_train, y_test):
        knn = KNeighborsClassifier()
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)
        print("Accuracy Score del KNN :", accuracy_score(y_test, y_pred) * 100, "%")
