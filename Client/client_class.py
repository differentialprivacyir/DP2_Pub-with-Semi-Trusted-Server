
import numpy as np
import random

 
class client_class:
    def __init__(self, epsilon, attributes_domain, client_data):
        self.epsilon = epsilon
        self.attributes_domain = attributes_domain
        self.client_data = client_data
        self.randomized_data = self.anonymizing()
    
     
    def anonymizing(self):
        perturbation_attributes = []
        eps = self.epsilon / len(self.attributes_domain.keys())
        for i in self.attributes_domain.keys():
            p = np.exp(eps)/(np.exp(eps)+len(self.attributes_domain.get(i))-1)
            q = (1-p)/(len(self.attributes_domain.get(i))-1)
            perturbation_attributes.append(self.generalized_randomize_response(self.client_data.get(i), self.attributes_domain.get(i), p))
        return perturbation_attributes


    def generalized_randomize_response(self, data, data_domain, p):
        coin = random.random()
        if coin <= p :
            return data
        else:
            return random.choice([ele for ele in data_domain if ele != data ])
           