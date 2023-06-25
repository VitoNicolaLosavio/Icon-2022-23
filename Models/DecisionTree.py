import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import validation_curve, RepeatedStratifiedKFold, GridSearchCV, learning_curve
from sklearn import tree

from sklearn.tree import DecisionTreeClassifier


class MyDecisionTreeClassifier:

    def __init__(self, x_train, x_test, y_train, y_test):
        self.dtree = DecisionTreeClassifier()
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def evaluation_model(self, seed, file_name: str):
        self.dtree.fit(self.x_train, self.y_train)
        tree_para = {'criterion': ['gini', 'entropy'], 'max_depth': np.arange(3, 15)}

        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=seed)
        grid = GridSearchCV(self.dtree, tree_para, cv=cv, scoring="accuracy", error_score=0)
        grid.fit(self.x_train, self.y_train)
        best_model: DecisionTreeClassifier = self.dtree.set_params(**grid.best_params_)
        best_model.fit(self.x_train, self.y_train)

        print('ALBERO RISULTANTE')
        tree.export_graphviz(best_model,
                             out_file='../TreeGraph/' + file_name, feature_names=self.x_train.columns.tolist(),
                             class_names=list(map(lambda x: str(x), self.y_train.unique().tolist())),
                             filled=True,
                             rounded=True
                             )

        tree.plot_tree(best_model,
                       feature_names=self.x_train.columns.tolist(),
                       class_names=list(map(lambda x: str(x), self.y_train.unique().tolist())),
                       filled=True,
                       rounded=True
                       )
        plt.show()
