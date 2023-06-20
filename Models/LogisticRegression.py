import sklearn
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, learning_curve
import numpy as np

# TODO sistema qui
class LogisticRegressionClass:

    def __init__(self, x_train, x_test, y_train, y_test, seed):
        log_reg = LogisticRegression(solver="saga", class_weight='balanced', penalty="l1",
                                     multi_class='multinomial', max_iter=2000, C=1,
                                     tol=1e-4, n_jobs=-1)

        log_reg.fit(x_train, y_train)
        y_pred_proba = log_reg.predict_proba(x_test)
        y_pred = log_reg.predict(x_test)


        C_val = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7]

        param_grid = dict(penalty=["l1", "l2"],
                          C=C_val,
                          )
        log_reg_GS = LogisticRegression(class_weight='balanced',
                                        multi_class='multinomial',
                                        solver="saga",
                                        max_iter=4000,
                                        tol=1e-4, n_jobs=-1)
        # Creo uno schema di cross validazione mischiata e stratificata grazie alla seguente funzione (voglio k=5):
        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=seed)
        RS = GridSearchCV(log_reg_GS, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1)
        search = RS.fit(x_train, y_train)
        bp = search.best_params_

        print(bp)

        # Preparo la partizione train-cross val:
        train_ind = np.ones(x_train.shape[0])
        train_ind = -train_ind
        val_ind = np.zeros(x_test.shape[0])
        ps = sklearn.model_selection.PredefinedSplit(test_fold=np.concatenate((train_ind, val_ind)))

        # La funzione vuole che passiamo train e val set assieme:
        X = np.concatenate((x_train, x_test), axis=0)
        y = np.concatenate((y_train, y_test))
        # cv mi permetterà di trattare gli esempi indicizzati come -1 come train e gli altri come validation set

        estimator = LogisticRegression(solver="saga", class_weight='balanced', penalty="l1",
                                       multi_class='multinomial', max_iter=2000, C=1,
                                       tol=1e-4, n_jobs=-1)

        # Ho creato una lista di indici con -1 per tutti gli indici del train e 0 per quelli del validation set.
        # Uso "learning_curve" di sklearn, specificando lo stimatore (ovvero il modello):
        train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=ps,
                                                                scoring="neg_log_loss",
                                                                train_sizes=np.linspace(0.1, 1.0, 50),
                                                                verbose=True,
                                                                n_jobs=-1)
        # Valutiamo l'andamento medio della curva sia su train che validation set:
        plt.subplots(1, figsize=(15, 10))
        plt.plot(train_sizes, train_scores, color="blue", label="Training set")
        plt.plot(train_sizes, test_scores, color="darkorange", label="Cross-validation set")

        plt.title("LEARNING CURVE\n(Regressione logistica)\n", color='red', fontsize=20)
        plt.xlabel("\nDimensioni del training set\n", color='black', fontsize=15)
        plt.ylabel("\nFunzione di costo\n", color='black', fontsize=15), plt.legend(loc="best")
        plt.grid()
        plt.show()


