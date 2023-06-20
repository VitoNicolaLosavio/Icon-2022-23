% Atom definitions
:-style_check(-singleton).
role(ResearchScientist).
role(Manager).
role(LaboratoryTechnician).

age(21).
age(22).
age(23).
age(24).
age(25).
age(26).
age(27).
age(28).
age(29).
age(30).
age(31).
age(32).
age(33).
age(34).
age(35).
age(36).
age(37).
age(38).

educationField(TechnicalDegree).
educationField(LifeSciences).

numCompanies(0).
numCompanies(1).
numCompanies(2).
numCompanies(3).
numCompanies(4).

businessTravel(TravelRarely).
businessTravel(TravelFrequently).
businessTravel(NonTravel).

%Role definitions
suitable(person(Role, Age, EducationField, NumCompanies, BusinessTravel)) :-
    role(Role),
    age(Age),
    educationField(EducationField),
    numCompanies(NumCompanies),
    businessTravel(BusinessTravel).
