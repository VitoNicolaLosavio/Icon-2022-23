from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


class KMeansClass:
    def __init__(self, x_train, x_test, y_train, y_test):
        k_means = KMeans()
        k_means.fit(x_train, y_train)
        y_pred = k_means.predict(x_test)
        # print("Precision del KMeans:", precision_score(y_test, y_pred, average='weighted'))
        # print("Recall del KMeans:", recall_score(y_test, y_pred, average='weighted'))
        # print("F1 del KMenas:", f1_score(y_test, y_pred, average='weighted'))
        # print("Accuracy Score del KMeans :", accuracy_score(y_test, y_pred) * 100, "%")
        print("Report del KMeans")
        print(classification_report(y_pred, y_test, zero_division=1))
