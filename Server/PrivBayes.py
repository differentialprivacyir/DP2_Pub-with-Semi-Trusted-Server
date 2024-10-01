
import random
import warnings
from itertools import combinations, product
from math import ceil
from multiprocessing.pool import Pool

import numpy as np
from scipy.optimize import fsolve

import random
from pandas import Series, DataFrame
from sklearn.metrics import mutual_info_score

"""
This module is based on PrivBayes in the following paper:

Zhang J, Cormode G, Procopiuc CM, Srivastava D, Xiao X.
PrivBayes: Private Data Release via Bayesian Networks.
"""

class PrivBayes:
    def __init__(self, data):
        self.data = data
        self.BN = self.greedy_bayes(data, 0)
    
    def set_random_seed(self, seed=0):
        random.seed(seed)
        np.random.seed(seed)

    def mutual_information(self, labels_x: Series, labels_y: DataFrame):
        """Mutual information of distributions in format of Series or DataFrame.

        Parameters
        ----------
        labels_x : Series
        labels_y : DataFrame
        """
        if labels_y.shape[1] == 1:
            labels_y = labels_y.iloc[:, 0]
        else:
            labels_y = labels_y.apply(lambda x: ' '.join(x.values), axis=1)

        return mutual_info_score(labels_x, labels_y)  

    def usefulness_minus_target(self, k, num_attributes, num_tuples, target_usefulness=5, epsilon=1):
        """Usefulness function in PrivBayes.

        Parameters
        ----------
        k : int
            Max number of degree in Bayesian networks construction
        num_attributes : int
            Number of attributes in dataset.
        num_tuples : int
            Number of tuples in dataset.
        target_usefulness : int or float
        epsilon : float
            Parameter of differential privacy.
        """
        if k == num_attributes:
            print('here')
            usefulness = target_usefulness
        else:
            usefulness = num_tuples * epsilon / ((num_attributes - k) * (2 ** (k + 3)))  # PrivBayes Lemma 3
        return usefulness - target_usefulness


    def calculate_k(self, num_attributes, num_tuples, target_usefulness=4, epsilon=1):
        """Calculate the maximum degree when constructing Bayesian networks. See PrivBayes Lemma 3."""
        default_k = 3
        initial_usefulness = self.usefulness_minus_target(default_k, num_attributes, num_tuples, 0, epsilon)
        if initial_usefulness > target_usefulness:
            return default_k
        else:
            arguments = (num_attributes, num_tuples, target_usefulness, epsilon)
            warnings.filterwarnings("error")
            try:
                ans = fsolve(self.usefulness_minus_target, np.array([int(num_attributes / 2)]), args=arguments)[0]
                ans = ceil(ans)
            except RuntimeWarning:
                print("Warning: k is not properly computed!")
                ans = default_k
            if ans < 1 or ans > num_attributes:
                ans = default_k
            return ans
        

    def worker(self, paras):
        child, V, num_parents, split, dataset = paras
        parents_pair_list = []
        mutual_info_list = []

        if split + num_parents - 1 < len(V):
            for other_parents in combinations(V[split + 1:], num_parents - 1):
                parents = list(other_parents)
                parents.append(V[split])
                parents_pair_list.append((child, parents))
                # TODO consider to change the computation of MI by combined integers instead of strings.
                mi = self.mutual_information(dataset[child], dataset[parents])
                mutual_info_list.append(mi)

        return parents_pair_list, mutual_info_list


    def greedy_bayes(self, dataset: DataFrame, k: int, seed=0):
        """Construct a Bayesian Network (BN) using greedy algorithm.

        Parameters
        ----------
        dataset : DataFrame
            Input dataset, which only contains categorical attributes.
        k : int
            Maximum degree of the constructed BN. If k=0, k is automatically calculated.
        epsilon : float
            Parameter of differential privacy.
        seed : int or float
            Seed for the randomness in BN generation.
        """
        self.set_random_seed(seed)
        dataset: DataFrame = dataset.astype(str, copy=False)
        num_tuples, num_attributes = dataset.shape
        if not k:
            k = self.calculate_k(num_attributes, num_tuples)
        print(k)
        print('================ Constructing Bayesian Network (BN) ================')
        root_attribute = random.choice(dataset.columns)
        V = [root_attribute]
        rest_attributes = list(dataset.columns)
        rest_attributes.remove(root_attribute)
        print(f'Adding ROOT {root_attribute}')
        N = []
        while rest_attributes:
            parents_pair_list = []
            mutual_info_list = []

            num_parents = min(len(V), k)
            tasks = [(child, V, num_parents, split, dataset) for child, split in
                    product(rest_attributes, range(len(V) - num_parents + 1))]
            with Pool() as pool:
                res_list = pool.map(self.worker, tasks)

            for res in res_list:
                parents_pair_list += res[0]
                mutual_info_list += res[1]

            idx = mutual_info_list.index(max(mutual_info_list))

            N.append(parents_pair_list[idx])
            adding_attribute = parents_pair_list[idx][0]
            V.append(adding_attribute)
            rest_attributes.remove(adding_attribute)
            print(f'Adding attribute {adding_attribute}')
        
        print('========================== BN constructed ==========================')
        print(N)


        return N

