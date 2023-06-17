import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pyswip import Prolog


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

    def valutation_features(self, orientation: str):
        """
        Funzione per la valutazione delle features di interesse
        tramite grafici

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
        Funzione usata per realizzare la feature target
        ci interconnettiamo al prolog per verificare regole
        e fatti definiti a livello aziendale
        :return:
        """
        prolog = Prolog()
        prolog.consult(sys.path[0] + '../Prolog_rules/rules.pl')
        print('sono qui')
        print(self.dataset.keys())
        for key, val in self.dataset.iterrows():
            target = bool(prolog.query(f"suitable(person({val['Role']},{val['Age']} ,{val['EducationField']} ,"
                                       f"{val['NumCompanies']} ,{val['BusinessTravel']} ))"))
            print(target)
