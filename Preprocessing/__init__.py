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
    #firstDataset.drop_columns(featureDropFirstDataset, "First_dataset.csv")
    #secondDataset.drop_columns(featureDropSecondDataset, "Second_dataset.csv")

    # Generazione dei grafici relativi alla valutazione delle features
    firstDataset.valutation_features('../Datasets/First_dataset.csv', "v")
    secondDataset.valutation_features('../Datasets/Second_dataset.csv', "h")

    #NOTE DA RIMUOVERE
    #TO DO:
    #Overrated in: MontlyInCome --> min 1009 max 20.0k
    #              NumCompaniesWorked --> min 0 max 9