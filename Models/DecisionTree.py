from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


class DecisionTree:

    def __init__(self, x_train, x_test, y_train, y_test):
        dtree = DecisionTreeClassifier()
        dtree.fit(x_train, y_train)
        y_pred = dtree.predict(x_test)
        print("Accuracy Score del Decision Tree:", accuracy_score(y_test, y_pred) * 100, "%")
