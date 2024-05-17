from collections import OrderedDict
import numpy as np
import random


class PRAM:
    def __init__(self, epsilon, clusters, PBC, attributes_domain, dataset):
        self.epsilon = epsilon
        self.clusters = clusters
        self.PBC = PBC
        self.attributes_domain = attributes_domain
        self.dataset = dataset
        self.randomized_data = self.purturbation()


    def purturbation(self):
        
        Q = dict()
        EPI = dict()
        for atr in self.attributes_domain.keys():

            ## first step --> Q
            cluster_index = self.cluster_detection(atr)
            e = self.epsilon * self.PBC[cluster_index]
            eps = e / len(self.clusters[cluster_index])
            p = (np.exp(eps)/(np.exp(eps)+len(self.attributes_domain.get(atr))-1))
            q = 1-p
            
            Q.update({atr:self.firstPurturbation(self.attributes_domain.get(atr), p, q)})


        ## LANDA
        LANDA = self.calculateLanda(self.dataset)

        for i in LANDA.keys():
            EPI.update({i:self.estimatePI(Q.get(i),LANDA.get(i))})
        ## second step --> Q'
        SQ = dict()
        for atr in self.attributes_domain.keys():
            SQ.update({atr:self.secondPurturbation(self.attributes_domain.get(atr), EPI.get(atr), Q.get(atr))})

        self.generalized_randomize_response(self.dataset,SQ)
            
        
    
    def cluster_detection(self, atr):
        for cl in self.clusters:
            if atr in cl:
                return self.clusters.index(cl)
        return -1
                  

    def firstPurturbation(self, domain, p, q):
        Q = []
        for d in range(len(domain)):
            attribute_Q = []
            for do in range(len(domain)):
                if d == do:
                    attribute_Q.append(p)
                else:
                    attribute_Q.append(q)
            Q.append(attribute_Q)
        return np.array(Q)

    
    def calculatePI(self, atr):
        values, counts = np.unique(self.dataset[[atr]], return_counts=True)
        values = list(values)
        o_counts = list(counts)
        atr_dict = dict()
        for i in range(len(values)):
            pi = o_counts[i] / len(self.dataset)
            atr_dict.update({values[i]:pi})
        ordered_dict = OrderedDict((key, atr_dict[key]) for key in self.attributes_domain.get(atr))
        return dict(ordered_dict)
    

    def calculateLanda(self, FPResult):
        LANDA = dict()
        for atr in FPResult.columns:
            values, counts = np.unique(FPResult[[atr]], return_counts=True)
            values = list(values)
            counts = list(counts)
            atr_dict = dict()
            for i in range(len(values)):
                pi = counts[i] / len(FPResult)
                atr_dict.update({values[i]:pi})
            ordered_dict = OrderedDict((key, atr_dict[key]) for key in self.attributes_domain.get(atr))
            LANDA.update({atr:dict(ordered_dict)})
        return LANDA
    

    def estimatePI(self, Q, Landa):
        Q_inverse = np.linalg.inv(Q)
        L = (np.array([list(Landa.values())])).transpose()
        estimate_pi = np.dot(Q_inverse, L)
        return estimate_pi
    

    def secondPurturbation(self, domain, EPI, Q):
        SQ = []
        for i in range(len(domain)):
            attribute_Q = []
            for j in range(len(domain)):
                a = EPI[j,0] * Q[j,i]
                s = 0
                for k in range(len(domain)):
                    s += (EPI[k,0] * Q[k,i])
                qbar = a / s
                if qbar < 0:
                    qbar = 0
                attribute_Q.append(qbar)
            SQ.append(attribute_Q)
        return SQ
    

    def generalized_randomize_response(self, dataset, SQ):
        for index in range(len(dataset)):
            for column in dataset.columns:
                coin = round(random.random(),4)
                Q_bar = SQ.get(column)
                domain = self.attributes_domain.get(column)
                iddd = domain.index(dataset[column].values[index])
                print(coin)
                print(Q_bar[iddd])
                print("::::::::::::::::::::::")
                p = min((number for number in Q_bar[iddd] if number >= coin), key=lambda x: abs(x - coin))
                print(p)
                idd = Q_bar[iddd].index(p)
                dataset.loc[index, [column]] = self.attributes_domain.get(column)[idd]
        return dataset 