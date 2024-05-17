import numpy as np
import math

class TVD:
    def __init__(self, original_data, perturbed_data, domains):
        self.original_data = original_data
        self.perturbed_data = perturbed_data
        self.domains = domains
        print(self.total_variation_distance())

    def total_variation_distance(self):
        """
        Computes the total variation distance between two (classical) probability
        measures P(x) and Q(x).
            tvd(P,Q) = (1/2) \sum_x |P(x) - Q(x)|
        """
        tvd = 0 
        for atr in self.original_data.columns:
            o_values, o_counts = np.unique(self.original_data[[atr]], return_counts=True)
            o_values = list(o_values)
            p_values, p_counts = np.unique(self.perturbed_data[[atr]], return_counts=True)
            p_values = list(p_values)
            atvd = 0 # attribute total variation distance
            for d in self.domains.get(atr):
                # index for o_values & p_values is same. so you can delete one of them
                index = o_values.index(d)
                atvd += abs(o_counts[index] - p_counts[index])
            atvd = atvd * (1/2)*(1/len(self.original_data))
            tvd += atvd
        tvd = tvd * (1/len(self.domains.keys()))
        print(tvd)
        return tvd