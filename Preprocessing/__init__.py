from ModellingDataset import Dataset

if __name__ == '__main__':
    # We use the same Dataset for the two cases of study
    firstDataset = Dataset('WA_Fn-UseC_-HR-Employee-Attrition.csv')
    secondDataset = Dataset('WA_Fn-UseC_-HR-Employee-Attrition.csv')

    featureDropFirstDataset = ['Age', 'BusinessTravel', 'EducationField', 'Gender',
                               'JobLevel', 'JobRole', 'NumCompaniesWorked']

    featureDropSecondDataset = ['BusinessTravel', 'DailyRate', 'DistanceFromHome',
                                'HourlyRate', 'JobInvolvement', 'JobSatisfaction',
                                'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'OverTime',
                                'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
                                'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
                                'YearsSinceLastPromotion', 'YearsWithCurrManager']

    firstDataset.drop_column(featureDropFirstDataset)
    secondDataset.drop_column(featureDropSecondDataset)