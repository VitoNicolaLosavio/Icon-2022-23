from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


class RandomForest:

    def __init__(self, x_train, x_test, y_train, y_test):
        rfc = RandomForestClassifier()
        rfc.fit(x_train, y_train)
        y_pred = rfc.predict(x_test)
        print("Accuracy Score della Random Forest :", accuracy_score(y_test, y_pred) * 100, "%")

