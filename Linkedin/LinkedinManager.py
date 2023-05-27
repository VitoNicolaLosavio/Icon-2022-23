from linkedin_api import linkedin
import pandas as pd


class LinkedinManager:
    """
    Classe utilizzata per connetterci a linkedin,
    prendere i dati e costruire il dataset per la
    ricerca dei dipendenti
    """

    def __init__(self, username: str, password: str):
        """
        costruttore di classe utilizzato per connetterci
        ad un utente linkedin esistente
        """
        try:
            self.username = username
            self.password = password
            self.linkedin: linkedin.Linkedin = linkedin.Linkedin(username=username, password=password)
            self.search_people()
        except Exception as e:
            print(e)
            self.linkedin: linkedin.Linkedin | None = None

    def retry_connection(self):
        """
        reistanzia la connessione con la classe linkedin
        :return:
        """
        try:

            self.linkedin = linkedin.Linkedin(username=self.username,password=self.password)
        except Exception as e:
            print('ECCEZIONE')
            print(e)

    def search_people(self):
        if(self.linkedin):
            #array associativo che conterrà i dati da caricare nel dataset
            allData = []
            allProfile = self.linkedin.search_people(limit=10)
            print(allProfile)
            for currentProfile in allProfile:
                data: dict = {}
                print('currentProfile')
                if 'name' in currentProfile:
                    data['Name'] = currentProfile['name']
                else:
                    data['Name'] = None
                if 'location' in currentProfile:
                    data['Location'] = currentProfile['location']
                else:
                    data['Location'] = None
                if 'jobtitle' in currentProfile:
                    data['Job_title'] = currentProfile['jobtitle']
                else:
                    data['Job_title'] = None
                info: dict = self.linkedin.get_profile(public_id=currentProfile['public_id'])
                if 'student' in info:
                    data['Student'] = info['student']
                else:
                    data['Student'] = None
                try:
                    data['Num_experience'] = list(info).count('experience')
                except:
                    data['Num_experience'] = None
                try:
                    data['Education'] = info['education'][0]['schoolName']
                except Exception as e:
                    data['Education'] = None
                try:
                    data['Field_of_study'] = info['education'][0]['fieldOfStudy']
                except Exception as e:
                    data['Field_of_study'] = None
                allData.append(data)
                print(allData)
            dataFrame = pd.DataFrame(allData)
            print(dataFrame)
        else:

            self.retry_connection()
