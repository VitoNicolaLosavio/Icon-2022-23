import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV


class RandomForest:

    def __init__(self, x_train, x_test, y_train, y_test):
        self.rfc = RandomForestClassifier()
        self.x_train: pd.DataFrame = x_train
        self.x_test: pd.DataFrame = x_test
        self.y_train: pd.DataFrame = y_train
        self.y_test: pd.DataFrame = y_test

    def evaluate_model(self, seed):
        param_dist = {'n_estimators': np.arange(3, 25),
                      'max_depth': np.arange(3, 15)}

        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=seed)
        grid = GridSearchCV(self.rfc, param_dist, cv=cv, scoring="accuracy", error_score=0)
        grid.fit(self.x_train, self.y_train)

        best_model: RandomForestClassifier = self.rfc.set_params(**grid.best_params_)
        best_model.fit(self.x_train, self.y_train)

        ax = plt.gca()
        RocCurveDisplay.from_estimator(self.rfc, self.x_test, self.y_test, ax=ax)

        plt.show()


