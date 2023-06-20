from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier


class NeuralNetwork:

    def __init__(self, x_train, x_test, y_train, y_test):
        neural_network = MLPClassifier(solver='lbfgs', alpha=1e-5,
                                       hidden_layer_sizes=(100, 2), random_state=1)
        neural_network.fit(x_train, y_train)
        y_pred = neural_network.predict(x_test)
        print("Accuracy Score della Neural Network :", accuracy_score(y_test, y_pred) * 100, "%")
