from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score


class KMeansClass:
    def __init__(self, x_train, x_test, y_train, y_test):
        k_means = KMeans()
        k_means.fit(x_train, y_train)
        y_pred = k_means.predict(x_test)
        print("Accuracy Score del KMeans :", accuracy_score(y_test, y_pred) * 100, "%")
