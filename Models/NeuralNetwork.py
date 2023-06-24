from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    RocCurveDisplay
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.neural_network import MLPClassifier


class NeuralNetwork:

    def __init__(self, x_train, x_test, y_train, y_test):
        self.neural_network = MLPClassifier()
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def evaluate_model(self, seed):
        self.neural_network.fit(self.x_train, self.y_train)
        param_grid  = {
            'hidden_layer_sizes': [ (10, ), (20,)],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam'],
            'alpha': [0.0001, 0.05],
            'learning_rate': ['constant', 'adaptive'],
        }

        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=seed)
        grid = GridSearchCV(self.neural_network, param_grid=param_grid, cv=cv, scoring="accuracy", error_score=0)
        grid.fit(self.x_train, self.y_train)
        best_model: MLPClassifier = self.neural_network.set_params(**grid.best_params_)

        best_model.fit(self.x_train, self.y_train)

        ax = plt.gca()
        RocCurveDisplay.from_estimator(best_model, self.x_test, self.y_test, ax=ax)

        plt.show()
