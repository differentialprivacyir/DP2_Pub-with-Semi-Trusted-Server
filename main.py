import copy
import numpy as np
import pandas as pd
from Clinet.client_class import client_class
from Server.PrivBayes import PrivBayes 
from Server.Clustering import clustering 
from Server.PRAM import PRAM
from Server.StatisticalAnalyses.TVD import TVD 

def read_dataset():
        df = pd.read_csv('FIMU_2017_Vhs.csv')
        del df['Person ID']
        del df['Name']
        return df

def attributes_domain(ds):
        atr_domain = dict()
        for column in ds.columns:
                dfnew = np.unique(copy.deepcopy(ds[column]))
                dfnew = dfnew.tolist()
                atr_domain.update({column:dfnew})
        return atr_domain

if __name__ == "__main__":
        dataset = read_dataset()
        domains = attributes_domain(dataset)
        clients_dataset = []

        for index, row in dataset.iterrows():
                clients_dataset.append(row.to_dict())

        epsilon = 0.8
        clients = []
        for data in clients_dataset:
                client = client_class(epsilon, domains, data)
                clients.append(client.randomized_data)
        df = pd.DataFrame(clients, columns = dataset.columns.to_list())
        pb = PrivBayes(df)
        clu = clustering(pb.BN, domains)

        P = PRAM(epsilon, clu.clusters, clu.PBC, domains, dataset)
        
 