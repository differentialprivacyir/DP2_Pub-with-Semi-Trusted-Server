
import itertools
import pandas as pd
import numpy as np



class PRAM:
    def __init__(self, epsilon, clusters, PBC, attributes_domain, dataset):
        self.epsilon = epsilon
        self.clusters = clusters
        self.PBC = PBC
        self.attributes_domain = attributes_domain
        self.dataset = dataset
        self.randomized_data = self.purturbation()

    def purturbation(self):
        print("========================= invariant PRAM ==========================")
        i = -1
        dataset_randomize = self.dataset.copy()

        for cl in self.clusters:
            i = i + 1
            Q = dict()
            print(cl)
            # Q calculation for each variable in cluster i-th
            for atr in cl:
                e = self.epsilon * self.PBC[i]
                eps = e / len(cl)
                p = (np.exp(eps)/(np.exp(eps)+len(self.attributes_domain.get(atr))-1))
                q = (1-p)/(len(self.attributes_domain.get(atr))-1)
                Q.update({atr:self.Q_calculation(self.attributes_domain.get(atr), p, q)})

            # Q calculation for Compound Variables
            Qatr1 = Q.get(cl[0])
            if len(cl)>1 :
                Qatr2 = Q.get(cl[1])
                Compound_Q = np.kron(Qatr1,Qatr2)
                for atr in cl[2:]:
                    Compound_Q = np.kron(Compound_Q, Q.get(atr))
            else:
                Compound_Q = Q.get(cl[0])


            ## LAMBDA calculation
            LAMBDA = self.lambda_Compound_calculation(cl, dataset_randomize)
            EPI = self.estimatePI(Compound_Q,LAMBDA)
            # π estimation

            # Q' calculation
            SQ = dict()
            SQ.update({'cl'+str(i):self.Q_bar_calculation(cl, EPI, Compound_Q)})

           
            # second GRR
            dataset_randomize = self.generalized_randomize_response(cl, dataset_randomize, SQ.get('cl'+str(i)))

        return dataset_randomize
    



    # Q calculation
    def Q_calculation(self, domain, p, q):
        Q = np.full((len(domain), len(domain)), q)
        np.fill_diagonal(Q, p)

        return Q



    # Lambda calculation(λ)
    def lambda_Compound_calculation(self, cl ,FPResult):
        LAMBDA = []
        d = []

        # calculate all possible combinations
        for at in cl:
            d.append(self.attributes_domain.get(at))
        combinations = list(itertools.product(*d))

        # calculate Lambda for each combination
        for combination in combinations:
            query_condition = pd.Series([True] * len(FPResult))
            for col, val in zip(cl, combination):
                query_condition &= (FPResult[col] == val)
            counts = query_condition.sum()
            combination_lambda = counts / len(FPResult)
            LAMBDA.append(combination_lambda)

        return LAMBDA



    # Π estimation (estimate of the original attribute variable distribution (Π = Q^-1 . λ))
    def estimatePI(self, Q, Landa):
        Q_inverse = np.linalg.inv(Q)
        #Q_inverse=self.matrix_inverse_normalization(Q)
        L = (np.array(Landa)).transpose()
        estimate_pi = np.dot(Q_inverse, L)
        for val in estimate_pi:
            if val < 0:
                regularized = estimate_pi - np.min(estimate_pi) + 1e-6  # Avoids division by zero or extreme scaling
                # Step 2: Normalize to sum to 1
                normalized = regularized / np.sum(regularized)
                return normalized
        return estimate_pi



    # Q invrse normalization
    def matrix_inverse_normalization(self, Q):
        Q_inverse = np.linalg.inv(Q)
        #return Q_inverse
        min_value = Q_inverse.min()
        A_shifted = Q_inverse - min_value + 1e-6
        row_sums = A_shifted.sum(axis=1, keepdims=True)
        A_row_normalized = A_shifted / row_sums
        A_row_normalized_rounded = np.round(A_row_normalized, decimals=15)
        return A_row_normalized_rounded



    # Q' calculation
    def Q_bar_calculation(self, cluster, EPI, CQ):

        combinations = 1
        for atr in cluster:
            combinations = combinations * len(self.attributes_domain.get(atr))
        SQ = []
        EPI_CQ = [[EPI[j] * CQ[j, i] for j in range(combinations)] for i in range(combinations)]
        denominators = [sum(EPI[k] * CQ[k, i] for k in range(combinations)) for i in range(combinations)]
        # Now iterate and calculate the qbar values
        for i in range(combinations):
            attribute_Q = []
            for j in range(combinations):
                a = EPI[j]*CQ[j,i]
                s = denominators[i]
                qbar = a / s if s != 0 else 0
                attribute_Q.append(qbar)
            SQ.append(attribute_Q)
        min_value = min([item for sublist in SQ for item in sublist])

        # Convert SQ to a numpy array for array operations
        SQ_array = np.array(SQ)

        # Shift the array by subtracting the minimum value
        A_shifted = SQ_array - min_value

        # Calculate row sums for normalization, keeping dimensions for broadcasting
        row_sums = A_shifted.sum(axis=1, keepdims=True)

        # Normalize each row by its row sum
        A_row_normalized = A_shifted / row_sums

        # Round the result to 15 decimal places
        A_row_normalized_rounded = np.round(A_row_normalized, decimals=15)

        return A_row_normalized_rounded
        #return SQ



    # apply GRR based on compound variable(Q or Q')
    def generalized_randomize_response(self, cluster, dataset, SQ):

        combinations_number = 1
        for atr in cluster:
            combinations_number = combinations_number * len(self.attributes_domain.get(atr))

        d = []
        for atr in cluster:
            d.append(self.attributes_domain.get(atr))

        combinations = list(itertools.product(*d))

        for u in range(len(dataset)):
            valuess = []
            for atr in cluster:
                valuess.append(dataset[atr].values[u])

            combination_index = combinations.index(tuple(valuess))

            combinations_list = [d for d in range(len(combinations))]
            random_index = np.random.choice(combinations_list , p=SQ[combination_index])

            k = -1
            for atr in cluster:
                k = k + 1
                dataset.loc[u, [atr]] =  combinations[random_index][k]

        return dataset

