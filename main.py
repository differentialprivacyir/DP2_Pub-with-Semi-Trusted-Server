import itertools
import pandas as pd

from Client.first_perturbation import first_perturbation
from Server.PrivBayes import PrivBayes
from Server.clustering import clustering
from Server.PRAM import PRAM
from Experimental_Evzluations.correlation import correlation
from Experimental_Evzluations.TVD import TVD




def read_dataset():
        input_domain='dataset/Data14-coarse.domain'
        input_data='dataset/Data14-coarse.dat'

        fd=open(input_domain,'r')
        head_line=fd.readline()
        readrow=head_line.split(" ")
        att_num=int(readrow[0])
        fd.close()

        ##########################################################################################
        #Get the attributes info in the domain file
        multilist = []
        fp=open(input_data,"r")
        fp.readline();  ###just for skip the header line


        while 1:
            line = fp.readline()
            if not line:
                break

            line=line.strip("\n")
            temp=line.split(",")
            temp2 = [str(i)+'a' for i in temp]
            multilist.append(temp2)
        fp.close()

        colomn = [str(i)+'a' for i in range(att_num)]
        df = pd.DataFrame(multilist, columns = colomn)
        return df

def attributes_domain():
        input_domain='dataset/Data14-coarse.domain'
        fd=open(input_domain,'r')
        head_line=fd.readline()
        readrow=head_line.split(" ")

        domain = dict()
        i=0
        while 1:
            line = fd.readline()
            if not line:
                break

            line=line.strip("\n")
            readrow=line.split(" ")
            start_x=0
            dom = []
            for eachit in readrow:
                start_x=start_x+1
                eachit.rstrip()
                if start_x>3:
                    dom.append(str(eachit)+'a')
            domain.update({str(i)+'a':dom})
            i=i+1
        fd.close()
        return domain

if __name__ == "__main__":
        dataset = read_dataset()
        domains = attributes_domain()
        _total_tvd_alpha_3_PRAM = 0

        _total_tvd_alpha_4_PRAM = 0

        _total_PRAM_correlation = 0

        total_epsilon = 7
        epsilon = total_epsilon
        print("epsilon : "+str(total_epsilon))

        for j in range(20):
            print("Loop number : "+ str(j+1))

            datasett = dataset.copy(deep=True)
            clients_dataset = dataset.copy(deep=True)
            clients_dataset2 = dataset.copy(deep=True)

            ## epsilon for each level is 2 but the total is 4
            

            clients_dataset_list = []
            for index, row in clients_dataset.iterrows():
                    clients_dataset_list.append(row.to_dict())

            clients = []
            for data in clients_dataset_list:
                client = first_perturbation(epsilon, domains, data)
                clients.append(client.randomized_data)
            FP = pd.DataFrame(clients, columns = dataset.columns.to_list())
            

            # constructing bayesian network  (k = 2)
            privBayes = PrivBayes(FP)

            # constructing clusters of size 3
            # len3 = False
            # while not len3:
            clu = clustering(privBayes.BN, domains, FP)
                # len3 = all(len(sublist) < 4 for sublist in clu.clusters)


            # invariant PRAM
            P = PRAM(epsilon, clu.clusters, clu.PBC, domains, FP)
            
            corr1 = correlation(datasett,P.randomized_data)
            print("correlation PRAM : "+ str(corr1.MAE))
            _total_PRAM_correlation += corr1.MAE

         
            #### alpha = 3
            _tvd_alpha_3_PRAM = 0
            subsets_of_size_3_2 = list(itertools.combinations(domains.keys(), 3))
            for subset in subsets_of_size_3_2:
                _tvd_alpha_3_PRAM += TVD(dataset[list(subset)], P.randomized_data[list(subset)] , domains).tvd
                

            _tvd_alpha_3_average_PRAM = _tvd_alpha_3_PRAM / len(subsets_of_size_3_2)
            _total_tvd_alpha_3_PRAM +=_tvd_alpha_3_average_PRAM
            #######################


            #### alpha = 4
            _tvd_alpha_4_PRAM = 0
           
            subsets_of_size_4_2 = list(itertools.combinations(domains.keys(), 4))
            for subset in subsets_of_size_4_2:
                _tvd_alpha_4_PRAM += TVD(dataset[list(subset)], P.randomized_data[list(subset)] , domains).tvd
              

            _tvd_alpha_4_average_PRAM = _tvd_alpha_4_PRAM / len(subsets_of_size_4_2)
            _total_tvd_alpha_4_PRAM +=_tvd_alpha_4_average_PRAM
          
            #######################


            print("alpah = 3 PRAM.... total variation distance : "+ str(_tvd_alpha_3_average_PRAM))
            
            print("alpah = 4 PRAM.... total variation distance : "+ str(_tvd_alpha_4_average_PRAM))
            print()
            


        _tvdd_3_PRAM = _total_tvd_alpha_3_PRAM / 20
       

        _tvdd_4_PRAM = _total_tvd_alpha_4_PRAM / 20
        

        total_correlation_PRAM = _total_PRAM_correlation/20
        

        print("PRAM.... average correlation : "+ str(total_correlation_PRAM))
       
        print("alpha = 3 PRAM.... average total variation distance : "+ str(_tvdd_3_PRAM))

        print("alpha = 4 PRAM.... average total variation distance : "+ str(_tvdd_4_PRAM))
       
