from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.neighbors import KNeighborsClassifier


class KNN:
    def __init__(self, x_train, x_test, y_train, y_test):
        knn = KNeighborsClassifier()
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)
        # print("Precision del KNN:", precision_score(y_test, y_pred))
        # print("Recall del KNN:", recall_score(y_test, y_pred))
        # print("F1 del KNN:", f1_score(y_test, y_pred))
        # print("Accuracy Score del KNN :", accuracy_score(y_test, y_pred) * 100, "%")
        print("Report del KNN")
        print(classification_report(y_pred, y_test))
