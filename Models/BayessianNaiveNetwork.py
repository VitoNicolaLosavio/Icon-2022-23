import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    RocCurveDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import BernoulliNB
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical


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
        y_pred = best_model.predict(self.x_test)

        print('REPORT DELLA MIGLIOR RETE NEURALE BAYESSIANA RITROVATA')
        print(classification_report(y_pred=y_pred, y_true=self.y_test))

