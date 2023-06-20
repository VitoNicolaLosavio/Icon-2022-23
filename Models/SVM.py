from sklearn import svm
from sklearn.metrics import accuracy_score


class SVM:

    def __init__(self,x_train ,x_test ,y_train, y_test):
        SVM = svm.SVC()
        SVM.fit(x_train, y_train)
        y_pred = SVM.predict(x_test)
        print("Accuracy Score della SVM è :", accuracy_score(y_test, y_pred) * 100, "%")
