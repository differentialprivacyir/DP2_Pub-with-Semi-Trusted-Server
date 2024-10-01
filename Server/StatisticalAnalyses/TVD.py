
###### TVD
class TVD:
    def __init__(self, original_data, perturbed_data, domains):
        self.original_data = original_data
        self.perturbed_data = perturbed_data
        self.domains = domains
        self.total_variation_distance()

    def total_variation_distance(self):
        # """
        # Computes the total variation distance between two (classical) probability
        # measures P(x) and Q(x).
        #     tvd(P,Q) = (1/2) \sum_x |P(x) - Q(x)|
        # """
        tvd = 0 
        for atr in self.original_data.columns:
            atvd = 0
            for d in self.domains.get(atr):
                o_count = 0
                p_count = 0
                for r in range(len(self.original_data)):
                    if str(self.original_data.at[r,atr]) == d:
                        o_count = o_count + 1
                    if str(self.perturbed_data.at[r,atr]) == d:
                        p_count = p_count + 1
                o_count = o_count / (len(self.original_data))
                p_count = p_count / (len(self.original_data))

                atvd += abs(o_count - p_count)
            atvd = atvd * (1/2)
            tvd += atvd
        tvd = tvd * (1/len(self.domains.keys()))
        print("total variation distance = " + str(tvd))
        

