import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, RocCurveDisplay
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV, learning_curve
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
        param_grid = {
            'hidden_layer_sizes': [(10, ), (20,)],
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

        score, train_scores, valid_scores = learning_curve(estimator=best_model,
                                                           X=self.x_train, y=self.y_train,
                                                           scoring='accuracy')

        mean_train_score = np.mean(train_scores, axis=1)

        mean_valuation_score = np.mean(valid_scores, axis=1)

        plt.title('curva di apprendimento')
        plt.plot(score, mean_train_score,
                 marker='o', markersize=5,
                 color='black', label='Training Accuracy')
        plt.plot(score, mean_valuation_score,
                 marker='o', markersize=5,
                 color='green', label='Validation Accuracy')
        plt.ylabel('Accuracy')
        plt.grid()
        plt.show()

        best_model.fit(self.x_train, self.y_train)
        y_pred = best_model.predict(self.x_test)

        print("REPORT DELLA MIGLIORE RETE NEURALE TROVATA")
        print(classification_report(y_pred, self.y_test))
