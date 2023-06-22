import sklearn
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, learning_curve
import numpy as np

class LogisticRegressionClass:

    def __init__(self, x_train, x_test, y_train, y_test, seed):
        logReg = LogisticRegression()
        logReg.fit(x_train, y_train)
        y_pred = logReg.predict(x_test)
        print("R2 Score della Regressione logistica :", r2_score(y_test, y_pred))
        print("MAE :", mean_absolute_error(y_test,y_pred))
        print("MSE :", mean_squared_error(y_test,y_pred))
        print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))


