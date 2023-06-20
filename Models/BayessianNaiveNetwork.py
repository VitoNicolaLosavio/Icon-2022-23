from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB


class GaussianNeuralBayes:
    def __init__(self, x_train, x_test, y_train, y_test):
        gaussian_nb = GaussianNB()
        gaussian_nb.fit(x_train, y_train)
        y_pred = gaussian_nb.predict(x_test)
        print("Accuracy Score della Rete neurale bayesiana :", accuracy_score(y_test, y_pred) * 100, "%")
