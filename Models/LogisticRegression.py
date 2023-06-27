import sklearn
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, learning_curve
import numpy as np


class LogisticRegressionClass:

    def __init__(self, x_train, x_test, y_train, y_test):
        self.log_reg = LogisticRegression()
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def evaluate_model(self, seed):
        C_val = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7]

        param_grid = dict(
            penalty=["l2"],
            C=C_val,
        )
        # Creo uno schema di cross validazione mischiata e stratificata grazie alla seguente funzione (voglio k=5):
        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=seed)
        clf = GridSearchCV(LogisticRegression(), param_grid=param_grid, cv=cv, verbose=True, n_jobs=1)

        clf.fit(self.x_train, self.y_train)
        best_model: LogisticRegression = self.log_reg.set_params(**clf.best_params_)
        best_model.fit(self.x_train, self.y_train)

        # Preparo la partizione train-cross val:
        train_ind = np.ones(self.x_train.shape[0])
        train_ind = -train_ind
        val_ind = np.zeros(self.x_test.shape[0])
        ps = sklearn.model_selection.PredefinedSplit(test_fold=np.concatenate((train_ind, val_ind)))

        # La funzione vuole che passiamo train e val set assieme:
        X = np.concatenate((self.x_train, self.x_test), axis=0)
        y = np.concatenate((self.y_train, self.y_test))
        # Ho creato una lista di indici con -1 per tutti gli indici del train e 0 per quelli del validation set.
        # Uso "learning_curve" di sklearn, specificando lo stimatore (ovvero il modello):
        train_sizes, train_scores, test_scores = learning_curve(best_model, X, y, cv=ps,
                                                                scoring="roc_auc",
                                                                train_sizes=np.linspace(0.1, 1.0, 50),
                                                                verbose=True,
                                                                n_jobs=-1)
        # Valutiamo l'andamento medio della curva sia su train che validation set:
        plt.plot(train_sizes, train_scores, color="blue", label="Training set")
        plt.plot(train_sizes, test_scores, color="darkorange", label="Cross-validation set")

        plt.title("LEARNING CURVE\n(Regressione logistica)\n", color='red', fontsize=20)
        plt.xlabel("\nDimensioni del training set\n", color='black', fontsize=15)
        plt.ylabel("\nFunzione di costo\n", color='black', fontsize=15), plt.legend(loc="best")
        plt.grid()
        plt.show()

        y_pred = best_model.predict(self.x_test)

        print("R2 Score della Regressione logistica :", abs(r2_score(self.y_test, y_pred)))
        print("MAE :", mean_absolute_error(self.y_test, y_pred))
        print("MSE :", mean_squared_error(self.y_test, y_pred))
        print("RMSE:", np.sqrt(mean_squared_error(self.y_test, y_pred)))

