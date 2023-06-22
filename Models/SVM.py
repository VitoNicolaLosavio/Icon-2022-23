from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report


class SVM:

    def __init__(self, x_train, x_test, y_train, y_test):
        SVM = svm.SVC()
        SVM.fit(x_train, y_train)
        y_pred = SVM.predict(x_test)
        # print("Precision della SVM:", precision_score(y_test, y_pred))
        # print("Recall della SVM:", recall_score(y_test, y_pred))
        # print("F1 della SVM:", f1_score(y_test, y_pred))
        # print("Accuracy Score della SVM:", accuracy_score(y_test, y_pred) * 100, "%")
        print("Report della SVM")
        print(classification_report(y_pred, y_test))
