% Atom definitions
role(researchScientist).
role(manager).
role(laboratoryTechnician).

age(21).
age(22).
age(23).
age(24).
age(25).

educationField(technical_Degree).
educationField(life_sciences).

numCompanies(0).
numCompanies(1).
numCompanies(2).
numCompanies(3).
numCompanies(4).

businessTravel(travel_Rarely).
businessTravel(travel_Frequently).
businessTravel(non_Travel).

% Rule definition
suitable(person(Role, Age, EducationField, NumCompanies, BusinessTravel)) :-
    role(Role),
    age(Age),
    educationField(EducationField),
    numCompanies(NumCompanies),
    businessTravel(BusinessTravel).
