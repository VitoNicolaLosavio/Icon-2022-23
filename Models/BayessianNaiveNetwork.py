import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, RocCurveDisplay
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.naive_bayes import BernoulliNB


class GaussianNeuralBayes:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.gaussian_nb = BernoulliNB()
        self.x_train: pd.DataFrame = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def evaluate_model(self, seed):
        params_grid = {'alpha': [0.01, 0.5, 1.0, 10.0, 20.0],
                       'fit_prior': [True, False],
                       'class_prior': [None, [0.1, ] * len(self.x_train.columns.tolist()), ],
                       'binarize': [None, 0.0, 8.5, 10.0]
                       }
        grid = GridSearchCV(self.gaussian_nb, param_grid=params_grid, cv=5, scoring="accuracy", error_score=0)

        grid.fit(self.x_train, self.y_train)

        best_model: BernoulliNB = self.gaussian_nb.set_params(**grid.best_params_)
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

        print('REPORT DELLA MIGLIOR RETE NEURALE BAYESSIANA RITROVATA')
        print(classification_report(y_pred=y_pred, y_true=self.y_test))

