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

        return self.generalized_randomize_response(self.dataset,SQ)
            
        
    
    def cluster_detection(self, atr):
        p_b_c = 2
        indexx = -1
        for cl in self.clusters:
            if atr in cl:
                a = self.PBC[self.clusters.index(cl)]
                if a < p_b_c:
                    p_b_c = a
                    indexx = self.clusters.index(cl)
         
        return indexx
                  

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
            atr_dict = dict()
            for d in self.attributes_domain.get(atr):
                counts = 0
                for r in range(len(FPResult)):
                    if str(FPResult.at[r,atr]) == d:
                        counts = counts + 1
                pi = counts / len(FPResult)
                atr_dict.update({d:pi})
            LANDA.update({atr:atr_dict})

        return LANDA
    
  
    

    def estimatePI(self, Q, Landa):
        Q_inverse=self.matrix_inverse_normalization(Q)
        L = (np.array([list(Landa.values())])).transpose()
        estimate_pi = np.dot(Q_inverse, L)
        return estimate_pi
    
    def matrix_inverse_normalization(self, Q):
        #print(Q)
        Q_inverse = np.linalg.inv(Q)
        #print(Q_inverse)
        min_value = Q_inverse.min()
        A_shifted = Q_inverse - min_value
        row_sums = A_shifted.sum(axis=1, keepdims=True)
        A_row_normalized = A_shifted / row_sums
        A_row_normalized_rounded = np.round(A_row_normalized, decimals=15)
        #print(A_row_normalized_rounded)
        return A_row_normalized_rounded

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
        #print(SQ)
        return SQ
    

    def generalized_randomize_response(self, dataset, SQ):
        for index in range(len(dataset)):
            for column in dataset.columns:
                #coin = random.random()
                Q_bar = SQ.get(column)
                domain = self.attributes_domain.get(column)
                iddd = domain.index(dataset[column].values[index])

                # print(iddd)
                # print("poiuytttttttttttttttttttttttttttttttttttttttttttttt")
                # ######################################################################################################################################################################################
                ### one method to choose randomly ---> it dosent give me a good result---> TVD > 0.3 just same as the TVD between original dataset and first purturbation
                # rand_num = np.random.rand()
                # cumulative_prob = np.cumsum(Q_bar[iddd])
                # randomized_response = np.where(rand_num <= cumulative_prob)[0][0]
                # a = self.attributes_domain.get(column)[randomized_response]
                # dataset.loc[index, [column]] = a
                # ######################################################################################################################################################################################
                
                # ######################################################################################################################################################################################
                ### second method and it isn't good at all ----> TVD(dataset, SP) > TVD(dataset, FP)    
                # if max(Q_bar[iddd]) >= coin :
                #     p = min((number for number in Q_bar[iddd] if number >= coin), key=lambda x: abs(x - coin))
                # else:
                #     p = max(Q_bar[iddd])
                # idd = Q_bar[iddd].index(p)
                # dataset.loc[index, [column]] = self.attributes_domain.get(column)[idd]
                # ######################################################################################################################################################################################
                
                dataset.loc[index, [column]] = np.random.choice(self.attributes_domain.get(column) , p=Q_bar[iddd])
        return dataset 
    
  