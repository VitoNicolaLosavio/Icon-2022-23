from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report
from sklearn.ensemble import RandomForestClassifier


class RandomForest:

    def __init__(self, x_train, x_test, y_train, y_test):
        rfc = RandomForestClassifier()
        rfc.fit(x_train, y_train)
        y_pred = rfc.predict(x_test)
        # print("Precision del Random Forest:", precision_score(y_test, y_pred))
        # print("Recall del Random Forest:", recall_score(y_test, y_pred))
        # print("F1 del Random Forest:", f1_score(y_test, y_pred))
        # print("Accuracy Score del Random Forest:", accuracy_score(y_test, y_pred) * 100, "%")
        print("Report del Random Forest")
        print(classification_report(y_pred, y_test))
