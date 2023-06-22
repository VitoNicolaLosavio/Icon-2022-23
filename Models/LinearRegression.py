import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


class RegressioneLineare:
    def __init__(self, x_train, x_test, y_train, y_test):
        model = LinearRegression()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print("R2 Score della Regressione lineare :", r2_score(y_test, y_pred))
        print("MAE :", mean_absolute_error(y_test,y_pred))
        print("MSE :", mean_squared_error(y_test,y_pred))
        print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))



