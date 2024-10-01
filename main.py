import pandas as pd
from Client.client_class import client_class
from Server.PrivBayes import PrivBayes 
from Server.Clustering import clustering 
from Server.PRAM import PRAM
from Server.StatisticalAnalyses.TVD import TVD 

def read_dataset():
        input_domain='Data2-coarse.domain'
        input_data='Data2-coarse.dat'
    
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
        df.to_csv("dataset.csv", index=False)
        return df



def attributes_domain():
        input_domain='Data2-coarse.domain'
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
            #print(readrow)
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
        clients_dataset = []

        for index, row in dataset.iterrows():
                clients_dataset.append(row.to_dict())

        epsilon = 1
        clients = []
        for data in clients_dataset:
                client = client_class(epsilon, domains, data)
                clients.append(client.randomized_data)
        fd = pd.DataFrame(clients, columns = dataset.columns.to_list())
        fd.to_csv("firstPerturbation.csv",index=False)
        pb = PrivBayes(fd)
        clu = clustering(pb.BN, domains)

        P1 = PRAM(epsilon, clu.clusters, clu.PBC, domains, fd)
        fp = pd.read_csv("firstPerturbation.csv")


        _tvd = TVD(dataset, fp , domains)
        _tvd1 = TVD(dataset, P1.randomized_data , domains)

        
        