from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from Preprocessing.ModellingDataset import Dataset
from Models.RandomForest import RandomForest
from Models.SVM import SVM
from Models.DecisionTree import DecisionTree
from Models.KNN import KNN
from Models.NeuralNetwork import NeuralNetwork
from Models.LogisticRegression import LogisticRegressionClass

if __name__ == '__main__':
    seed = 53

    # Verrà usato lo stesso dataset per due diversi task
    firstDataset = Dataset('../Datasets/WA_Fn-UseC_-HR-Employee-Attrition.csv')
    secondDataset = Dataset('../Datasets/WA_Fn-UseC_-HR-Employee-Attrition.csv')

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

    # firstDataset.categorical_var_normalization("Normalized_FirstDataset.csv")
    # secondDataset.categorical_var_normalization("Normalized_SecondDataset.csv")

    normal_firstDataset = Dataset('../Datasets/Normalized_FirstDataset.csv')
    normal_secondDataset = Dataset('../Datasets/Normalized_SecondDataset.csv')

    # per il primo dataset
    x_train, x_test = train_test_split(normal_firstDataset.dataset,
                                       stratify=normal_firstDataset.dataset['Suitable'],
                                       test_size=0.30,
                                       train_size=0.70,
                                       shuffle=True, random_state=seed)
    x_train = shuffle(x_train)
    y_train = x_train['Suitable']
    x_train = x_train.drop('Suitable', axis=1)

    y_test = x_test['Suitable']
    x_test = x_test.drop('Suitable', axis=1)

    print('\nRISULTATI OTTENUTI DAL PRIMO DATASET\n')
    RandomForest(x_train, x_test, y_train, y_test)
    DecisionTree(x_train, x_test, y_train, y_test)
    SVM(x_train, x_test, y_train, y_test)
    KNN(x_train, x_test, y_train, y_test)
    NeuralNetwork(x_train, x_test, y_train, y_test)
    #LogisticRegressionClass(x_train, x_test, y_train, y_test, seed)


    # per il secondo dataset
    x_train, x_test = train_test_split(normal_secondDataset.dataset,
                                       stratify=normal_secondDataset.dataset['JobSatisfaction'],
                                       test_size=0.30,
                                       train_size=0.70,
                                       shuffle=True, random_state=seed)
    x_train = shuffle(x_train)
    y_train = x_train['JobSatisfaction']
    x_train = x_train.drop('JobSatisfaction', axis=1)

    y_test = x_test['JobSatisfaction']
    x_test = x_test.drop('JobSatisfaction', axis=1)

    print('\nRISULTATI OTTENUTI DAL SECONDO DATASET\n')
    RandomForest(x_train, x_test, y_train, y_test)
    DecisionTree(x_train, x_test, y_train, y_test)
    SVM(x_train, x_test, y_train, y_test)
    KNN(x_train, x_test, y_train, y_test)
    NeuralNetwork(x_train, x_test, y_train, y_test)
    #LogisticRegressionClass(x_train, x_test, y_train, y_test, seed)
