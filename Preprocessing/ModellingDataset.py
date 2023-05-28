import pandas as pd


class Dataset:
    """
    classe usata per modellare i dataaset
    """

    def __init__(self, dataset: str):
        """
        inizializzatore di classe con in input
        il dataset di cui fare il preprocesssing
        """
        self.dataset = pd.read_csv(dataset)
        print(self.dataset.shape)
        print(self.dataset.head(5))

    def drop_column(self, column: list):
        """
        Funzione che droppa le feature al momento inutili e
        verifichiamo che il dataset non contenga dati None
        """
        keys = self.dataset.keys()
        keys = keys.drop(column)
        self.dataset = self.dataset.drop(keys, axis=1)
        print(self.dataset.shape)
        print(self.dataset.head(5))
        # verify if there's a none value in dataset
        print(self.dataset.isnull().sum())

