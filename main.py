import sys

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from Models.MarkovChain import MarkovChain
from Preprocessing.ModellingDataset import Dataset
from Models.RandomForest import RandomForest
from Models.SVM import SVM
from Models.DecisionTree import MyDecisionTreeClassifier
from Models.KNN import KNN
from Models.NeuralNetwork import NeuralNetwork
from Models.BayessianNaiveNetwork import GaussianNeuralBayes
from Models.LinearRegression import RegressioneLineare
from Models.LogisticRegression import LogisticRegressionClass
from Models.Clustering import Clustering

if __name__ == '__main__':
    seed = 53

    print(sys.path[0])
    # Verrà usato lo stesso dataset per due diversi task
    firstDataset = Dataset('/Datasets/WA_Fn-UseC_-HR-Employee-Attrition.csv')
    secondDataset = Dataset('/Datasets/WA_Fn-UseC_-HR-Employee-Attrition.csv')

    # Features da mantere per il primo task
    featureDropFirstDataset = ['Age', 'BusinessTravel', 'EducationField', 'Gender',
                               'JobLevel', 'JobRole', 'NumCompaniesWorked']

    # Features da mantenere per il secondo task
    featureDropSecondDataset = ['BusinessTravel', 'DailyRate', 'DistanceFromHome',
                                'HourlyRate', 'JobInvolvement', 'JobSatisfaction',
                                'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'OverTime',
                                'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
                                'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
                                'YearsSinceLastPromotion', 'YearsWithCurrManager']

    # Drop delle features che non servono
    firstDataset.drop_columns(featureDropFirstDataset, "First_dataset.csv")
    secondDataset.drop_columns(featureDropSecondDataset, "Second_dataset.csv")

    # Generazione dei grafici relativi alla valutazione delle features
    # firstDataset.valutation_features("h")
    # secondDataset.valutation_features("h")

    # firstDataset.create_feature_target()


    secondDataset.numeric_variables()

    #firstDataset.categorical_var_normalization("/Normalized_FirstDataset.csv")
    secondDataset.categorical_var_normalization("/Normalized_SecondDataset.csv")


    normal_firstDataset = Dataset('/Datasets/Normalized_FirstDataset.csv')
    normal_secondDataset = Dataset('/Datasets/Normalized_SecondDataset.csv')

    # per il primo dataset
    X = normal_firstDataset.dataset
    Y = X['Suitable']
    X.drop('Suitable', axis=1)

    x_train, x_test, y_train, y_test = train_test_split(X, Y,
                                       stratify=Y,
                                       test_size=0.30,
                                       train_size=0.70,
                                       shuffle=True, random_state=seed)

    print('\nRISULTATI OTTENUTI DAL PRIMO DATASET')
    print('PER LA PRIMA PARTE UTILIZZIAMO DEI CLASSIFICATORI\n')
    RandomForest(x_train, x_test, y_train, y_test)
    MyDecisionTreeClassifier(x_train, x_test, y_train, y_test)
    SVM(x_train, x_test, y_train, y_test)
    KNN(x_train, x_test, y_train, y_test)
    NeuralNetwork(x_train, x_test, y_train, y_test)
    GaussianNeuralBayes(x_train, x_test, y_train, y_test)


    # per il secondo dataset
    X = normal_secondDataset.dataset
    Y = X['JobSatisfaction']
    X.drop('JobSatisfaction', axis=1)

    x_train, x_test, y_train_reg, y_test_reg = train_test_split(X, Y,
                                       stratify=round(Y),
                                       test_size=0.30,
                                       train_size=0.70,
                                       shuffle=True, random_state=seed)

    y_train = round(y_train_reg)
    y_test = round(y_test)
    print('\nRISULTATI OTTENUTI DAL SECONDO DATASET')
    print('PER LA SECONDA PARTE UTILIZZIAMO DEI CLASSIFICATORI\n')
    RandomForest(x_train, x_test, y_train, y_test)
    MyDecisionTreeClassifier(x_train, x_test, y_train, y_test)
    SVM(x_train, x_test, y_train, y_test)
    KNN(x_train, x_test, y_train, y_test)
    NeuralNetwork(x_train, x_test, y_train, y_test)
    GaussianNeuralBayes(x_train, x_test, y_train, y_test)
    RegressioneLineare(x_train, x_test, y_train_reg, y_test_reg)
    LogisticRegressionClass(x_train, x_test, y_train, y_test, seed)

    X = Dataset('/Datasets/WA_Fn-UseC_-HR-Employee-Attrition.csv')
    X.categorical_variable_for_all_dataset('dataset_normalized_for_unsupervisionated_learning.csv')
    X = Dataset('/Datasets/dataset_normalized_for_unsupervisionated_learning.csv')
    MarkovChain(X.dataset)
    Clustering(X.dataset, X.dataset.keys().__len__())
