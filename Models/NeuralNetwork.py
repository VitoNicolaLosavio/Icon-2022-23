from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.neural_network import MLPClassifier


class NeuralNetwork:

    def __init__(self, x_train, x_test, y_train, y_test):
        neural_network = MLPClassifier(solver='lbfgs', alpha=1e-5,
                                       hidden_layer_sizes=(100, 2), random_state=1)
        neural_network.fit(x_train, y_train)
        y_pred = neural_network.predict(x_test)
        # print("Precision della Rete neurale:", precision_score(y_test, y_pred, average='weighted'))
        # print("Recall della Rete neurale:", recall_score(y_test, y_pred, average='weighted'))
        # print("F1 della Rete neurale:", f1_score(y_test, y_pred, average='weighted'))
        # print("Accuracy Score della Rete neurale :", accuracy_score(y_test, y_pred) * 100, "%")
        print("Report della Rete neurale neurale")
        print(classification_report(y_pred, y_test))
