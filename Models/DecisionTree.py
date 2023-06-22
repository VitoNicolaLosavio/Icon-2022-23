from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.tree import DecisionTreeClassifier


class DecisionTree:

    def __init__(self, x_train, x_test, y_train, y_test):
        dtree = DecisionTreeClassifier()
        dtree.fit(x_train, y_train)
        y_pred = dtree.predict(x_test)
        #print("Precision del Decision Tree:", precision_score(y_test, y_pred))
        # print("Recall del Decision Tree:", recall_score(y_test, y_pred))
        # print("F1 del Decision Tree:", f1_score(y_test, y_pred))
        # print("Accuracy Score del Decision Tree:", accuracy_score(y_test, y_pred) * 100, "%")
        print("Report del Decision Tree")
        print(classification_report(y_pred, y_test))
