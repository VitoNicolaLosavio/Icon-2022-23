from ModellingDataset import Dataset

if __name__ == '__main__':
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
    #firstDataset.valutation_features("h")
    #secondDataset.valutation_features("h")


    #firstDataset.create_feature_target()

    firstDataset.create_feature_target()

    firstDataset.categorical_var_normalization("Normalized_FirstDataset.csv")
    secondDataset.categorical_var_normalization("Normalized_SecondDataset.csv")

    normal_firstDataset = Dataset('../Datasets/Normalized_FirstDataset.csv')
    normal_secondDataset = Dataset('../Datasets/Normalized_SecondDataset.csv')

    #print('VERIFICA PRIMO DATASET\n')
    #normal_firstDataset.count_variables()
    #print('VERIFICA SECONDO DATASET\n')
    #normal_secondDataset.count_variables()
