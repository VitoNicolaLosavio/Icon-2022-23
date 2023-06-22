from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.naive_bayes import GaussianNB


class GaussianNeuralBayes:
    def __init__(self, x_train, x_test, y_train, y_test):
        gaussian_nb = GaussianNB()
        gaussian_nb.fit(x_train, y_train)
        y_pred = gaussian_nb.predict(x_test)
        # print("Precision della Rete neurale bayesiana :", precision_score(y_test, y_pred))
        # print("Recall della Rete neurale bayesiana :", recall_score(y_test, y_pred))
        # print("F1 della Rete neurale bayesiana :", f1_score(y_test, y_pred))
        # print("Accuracy Score della Rete neurale bayesiana :", accuracy_score(y_test, y_pred) * 100, "%")
        print("Report della Rete neurale bayesian")
        print(classification_report(y_pred, y_test))
