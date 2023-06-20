import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pyswip import Prolog
import numpy as np


class Dataset:
    """
    Classe per la modellazione del dataset
    """

    def __init__(self, dataset: str):
        """
        Costruttore di classe.

        Parametri
        ----------
        dataset : nome del dataset su cui fare il preprocessing

        """
        self.dataset = pd.read_csv(dataset)
        print(self.dataset.shape)
        print(self.dataset.head(5))
        # dataFrame = sns.load_dataset(dataset)
        # sns.heatmap(dataFrame.corr(), annot=True)

    def drop_columns(self, columns: list, file_newDataset: str):
        """
        Funzione per l'eliminazione delle feature non di nostro interesse
        e la verifica di ulteriori dati a None

        Parametri
        ----------
        columns : lista dei nomi delle colonne da mantenere
        file_newDataset : stringa contenente il nome che si desidera dare al nuovo file csv
        """
        keys = self.dataset.keys()
        keys = keys.drop(columns)
        self.dataset = self.dataset.drop(keys, axis=1)
        print(self.dataset.shape)
        print(self.dataset.head(5))
        # Verifica della presenza di valori None nel dataset
        print(self.dataset.isnull().sum())
        # Comando usato una volta per splittare il dataset in due parti
        self.write_csv(file_newDataset)

    def write_csv(self, file_name: str):
        """
        Funzione per la creazione di un nuovo file csv

        Parameters
         ----------
        file_name : stringa contenente il nome che si desidera dare al file csv

        """
        rows = self.dataset.to_numpy()  # return: list of array for each columns
        df = pd.DataFrame(rows)
        df.columns = self.dataset.keys()
        df.to_csv("../Datasets/" + file_name, index=False)

    def create_boxplot(self, orientation: str):
        """
        Funzione per la creazione dei boxplot

        Parameters
        ----------
        orientation : orientamento del grafico (verticale,orizzontale)

        """
        # Apply the default theme
        sns.set_theme()
        # Load dataset
        # Create a visualizzation
        sns.catplot(data=self.dataset, kind="box", orient=orientation)
        plt.show()

    def create_feature_target(self):
        """
        Funzione per la realizzazione della feature target con l'interconnessione
        al prolog per la verifica delle regole e dei fatti definiti a livello aziendale

        """
        prolog = Prolog()
        prolog.consult("rules.pl")
        result = []

        for _, val in self.dataset.iterrows():
            query = f"suitable(person({str(val['JobRole']).replace(' ','')},{val['Age']},{str(val['EducationField']).replace(' ','')},{val['NumCompaniesWorked']},{str(val['BusinessTravel']).replace('_','').replace('-','')}))"
            result.append(bool(list(prolog.query(query))))

        self.dataset['Suitable'] = result
        self.write_csv("Complete_first_dataset.csv")

    def count_variables(self):
        """
        Funzione per il conteggio delle variabili
        """
        keys = self.dataset.keys()
        for i in keys:
            print(self.dataset[i].value_counts())



    def categorical_var_normalization(self, file_name:str):
        """
        Funzione per la normalizzazione delle variabili categoriche
        :param file_name: nome del nuovo file csv
        """
        allKeys = self.dataset.keys()

        for key in allKeys:
            if key == 'BusinessTravel':
                self.dataset[key] = self.dataset[key].replace('Travel_Rarely', 1)
                self.dataset[key] = self.dataset[key].replace('Travel_Frequently', 2)
                self.dataset[key] = self.dataset[key].replace('Non-Travel', 0)
            if key == 'Gender':
                self.dataset[key] = self.dataset[key].replace('Male', 1)
                self.dataset[key] = self.dataset[key].replace('Female', 0)
            if key == 'EducationField':
                all_possible_values = self.dataset[key].unique().tolist()
                all_possible_values.remove('Life Sciences')
                all_possible_values.remove('Technical Degree')
                self.dataset[key] = self.dataset[key].replace(['Life Sciences', 'Technical Degree'], 1)
                self.dataset[key] = self.dataset[key].replace(all_possible_values, 0)
            if key == "JobRole":
                all_possible_values = self.dataset[key].unique().tolist()
                all_possible_values.remove('Research Scientist')
                all_possible_values.remove('Manager')
                all_possible_values.remove('Laboratory Technician')
                self.dataset[key] = self.dataset[key].replace(['Research Scientist', 'Manager', 'Laboratory Technician'], 1)
                self.dataset[key] = self.dataset[key].replace(all_possible_values, 0)
            if key == "MaritalStatus":
                self.dataset[key] = self.dataset[key].replace('Single', 0)
                self.dataset[key] = self.dataset[key].replace('Married', 1)
                self.dataset[key] = self.dataset[key].replace('Divorced', 2)
            if key == "OverTime":
                self.dataset[key] = self.dataset[key].replace('Yes', 1)
                self.dataset[key] = self.dataset[key].replace('No', 0)
            if key == "Suitable":
                self.dataset[key] = self.dataset[key].replace(False, 0)
                self.dataset[key] = self.dataset[key].replace(True, 1)

        self.write_csv(file_name)
