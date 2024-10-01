# DP2-Pub


![photo](/img/Dp2_Pub_semiTrusted.png)

### main.py

برنامه از کلاس main.py شروع می‌شود.

در قطعه کد زیر کتابخانه‌ها و کلاس‌های موردنیاز فراخوانی می‌شود:

``` python
import pandas as pd
from data_generator.first_dataset import first_dataset
from Client.first_perturbation import first_perturbation
from Server.BayesianNetwork import BayesianNetwork 
from Server.Clustering import clustering 
from Client.second_perturbation import second_perturbation
from Server.StatisticalAnalyses.TVD import TVD 
from Server.StatisticalAnalyses.MSE import MSE
```
سپس مشخصات مجموعه‌داده مشخص می‌شود:

تعداد ویژگی‌ها:

```python
attributes_number = 100 
```

اندازه دامنه هر ویژگی به صورت تصادفی از بین بازه [s_domain_number و e_domain_number] انتخاب می‌شود:

```python
s_domain_number = 100 
e_domain_number = 150
```

تعداد کاربران:

```python
record_number = 1000
```

با استفاده از دستور زیر کلاس first_dataset برای تولید یک مجموعه‌داده عددی صدا زده می‌شود:

```python
# generate new dataset
f_dataset = first_dataset(attributes_number, s_domain_number, e_domain_number, record_number) 
```

### first_dataset.py

در کلاس first_dataset کتابخانه‌های زیر به کار گرفته شده است:

```python
import pandas as pd
import random
import numpy as np
```

مقداردهی اولیه متغیرها صورت گرفته و سپس تابع dataGenerator فراخوانی می‌شود:

```python
    
class first_dataset:
    def __init__(self, attributes_number, s_domain_number, e_domain_number, record_number):
        self.attributes_number = attributes_number
        self.s_domain_number = s_domain_number
        self.e_domain_number = e_domain_number
        self.record_number = record_number
        self.dataset, self.domains = self.dataGenerator()

```

خلاصه: در این تابع مقادیر هر ویژگی برای کاربران مشخص شده و دامنه هر یک از ویژگی‌ها نیز مشخص می‌شود.

جزییات: در تابع dataGenerator برای هر ویژگی یک اندازه دامنه بین s_domain_number و e_domain_number به صورت تصادفی انتخاب شده و سپس به تعداد record_number که تعداد کاربران است، عدد تصادفی بین ۱ و اندازه دامنه محاسبه شده، تولید می‌شود. همچنین دامنه‌های هر ویژگی، به متغیر atr_domain اضافه می‌شود(از این متغیر برای عملیات GRR استفاده می‌شود):

```python
   def dataGenerator(self):
        dataSet = dict()
        atr_domain = dict()
        for i in range(self.attributes_number):
            domain_number = random.randrange(self.s_domain_number, self.e_domain_number+1)
            s = list(np.random.randint(low = 1,high=domain_number+1,size=self.record_number))
            dataSet.update({str(i) : self.changetostring(s)})
            ss = list(range(1, domain_number+1))
            atr_domain.update({str(i):self.changetostring(ss)})
        return pd.DataFrame(dataSet), atr_domain 

    def changetostring(self, arr):
        array = []
        for ii in arr:
            array.append(str(ii))
        return array
```

### main.py

برای راحت‌تر انجام شدن مرحله بعد، هر یک از سطرهای مجموعه داده تولیدی را به یک dict تبدیل کرده و در یک لیست ذخیره می‌کنیم:

```python
# convert each row of dataframe(dataset) to dict
clients_dataset = []
for index, row in f_dataset.dataset.iterrows():
    clients_dataset.append(row.to_dict())
```

مقادیر بودجه حریم خصوصی تعیین می شود:

```python
# privacy budgets
 epsilon = 1
```


مراحل راهکار به شرح زیر می‌باشد: 

### مرحله اول:

هر یک از کاربران داده‌های خود را نوفه‌دار می‌کنند. به همین دلیل کلاس first_perturbation به ازای داده‌های هر کاربر فراخوانی شده و خروجی‌ آن که داده‌های نوفه‌دار است، به لیستی به نام clients اضافه می‌شود. و سپس لیست درست شده از داده‌های نوفه‌دار همه کاربران به dataframe تبدیل می‌شود.(این تبدیل برای این است که ورودی مرحله دوم dataframe است.)

```python
    # client side
    # first perturbation
    clients = []
    for data in clients_dataset:
        client = first_perturbation(e, f_dataset.domains, data)
        clients.append(client.randomized_data)

    # convert list to dataframe
    df = pd.DataFrame(clients, columns = f_dataset.dataset.columns.to_list())
```

### first_perturbation.py

در این کلاس از کتابخانه‌های زیر استفاده می‌شود:

```python
import numpy as np
import random
```

مقداردهی اولیه انجام گرفته و تابع anonymizing فراخوانی می‌شود:

```python
class first_perturbation:
    def __init__(self, epsilon, attributes_domain, client_data):
        self.epsilon = epsilon
        self.attributes_domain = attributes_domain
        self.client_data = client_data
        self.randomized_data = self.anonymizing()
```

در تابع anonymizing بودجه حریم خصوصی تقسیم بر تعداد ویژگی‌ها شده تا به هر ویژگی بودجه حریم خصوصی مساوی اختصاص یابد و سپس p , q محاسبه می‌شود و تابع generalized_randomize_response برای تصادفی سازی هر یک از ویژگی‌ها فراخوانی می‌شود و خروجی آن در لیست perturbation_attributes ذخیره می‌شود:

```python
    def anonymizing(self):
        perturbation_attributes = []
        eps = self.epsilon / len(self.attributes_domain.keys())
        for i in self.attributes_domain.keys():
            p = np.exp(eps)/(np.exp(eps)+len(self.attributes_domain.get(i))-1)
            # q = (1-p)/(len(self.attributes_domain.get(i))-1)
            perturbation_attributes.append(self.generalized_randomize_response(self.client_data.get(i), self.attributes_domain.get(i), p))
        return perturbation_attributes
```

 ورودی تابع generalized_randomize_response شامل داده‌ای که قرار است تصادفی شود به همراه دامنه‌های ممکن و p است. در این تابع یک عدد تصادفی بین 0 و 1 انتخاب می‌شود و اگر کوچکتر از مقدار p باشد، داده تغییر نمی‌کند، در غیر این‌صورت از بین دامنه‌های این ویژگی یک مقدار دیگر مخالف با مقدار کنونی به صورت تصادفی انتخاب می‌شود: 

```python
    def generalized_randomize_response(self, data, data_domain, p):
        coin = random.random()
        if coin <= p :
            return data
        else:
            return random.choice([ele for ele in data_domain if ele != data ])
           
```

### main.py

### مرحله دوم

در این مرحله که در سمت سرور انجام می‌گیرد، داده‌های نوفه‌دار تولید شده توسط کاربران به کلاس BayesianNetwork ارسال می‌شوند

```python
        # server side
        # generate bayesian network (PrivBayes)
        bn = BayesianNetwork(df,3)
```

### BayesianNetwork.py

این کلاس از پروژه موجود در 
[این آدرس](https://github.com/DataResponsibly/DataSynthesizer/blob/master/DataSynthesizer/lib/PrivBayes.py) گرفته شده است و فقط موارد اضافی حذف و چند تابع از
 [کلاس](https://github.com/DataResponsibly/DataSynthesizer/blob/master/DataSynthesizer/lib/utils.py)
  دیگر در همین پروژه به آن اضافه شده است.

در کلاس  BayesianNetwork از کتابخانه‌های زیر استفاده شده است: 

```python
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
```

مقداردهی اولیه صورت گرفته و تابع greedy_bayes فراخوانی می‌شود:

```python
class BayesianNetwork:
    def __init__(self, data, k):
        self.data = data
        self.k = k
        self.BN = self.greedy_bayes(data, k)
```

ابتدا یک ویژگی به صورت تصادفی انتخاب  و به متغیر V که لیستی از ویژگی‌های اضافه شده به شبکه بیزی در حال تولید است، اضافه می‌شود.  برای ساخت شبکه بیزی، حداکثر تعداد پدرانی که یک ویژگی می‌توانند داشته باشد با فرمول num_parents = min(len(V), k) مشخص می‌شود. سپس با فراخوانی تابع worker مجموعه پدران یک ویژگی به همراه اطلاعات متقابلی که بین آن ویژگی و مجمومه پدرانش وجود دارد، محاسبه شده و درنهایت ویژگی‌ای که اطلاعات متقابل محاسبه شده بیشتری داشته باشد، انتخاب شده و به متغیر V اضافه می شود و از لیست rest_attributes که لیست ویژگی‌های اضافه نشده به شبکه بیزی است، حذف می‌شود. اینکار را تا جایی که rest_attributes خالی شود(همه ویژگی‌ها انتخاب شوند) ادامه می‌دهیم.

```python
    def greedy_bayes(self, dataset: DataFrame, k: int, seed=0):
       
        self.set_random_seed(seed)
        dataset: DataFrame = dataset.astype(str, copy=False)
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
```

متغیر N نیز همان شبکه بیزی است.


تابع greedy_bayes که در تابع set_random_seed از آن استفاده شده است:

```python    
    def set_random_seed(self, seed=0):
        random.seed(seed)
        np.random.seed(seed)
```

تابع worker که  در تابع greedy_bayes از آن استفاده شده است:

```python
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
```

تابع mutual_information که در تابع worker از آن استفاده شده است:

```python
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
```

### main.py

### مرحله سوم

بعد از ایجاد شبکه بیزی، عملیات خوشه‌بندی انجام می‌گیرد به همین دلیل کلاس clustering فراخوانی می‌شود:

```python
        # server side
        # clustering
        clu = clustering(bn.BN, f_dataset.domains)
```

### clustering.py

در این کلاس از کتابخانه‌های زیر استفاده شده است:

```python
import math
import random
```

به طور خلاصه می‌توان گفت که در این کلاس ابتدا با فراخوانی تابع Attribute_Clustering خوشه‌بندی انجام می‌گیرد و سپس با فراخوانی تابع privacy_budget_coefficient درجه اهمیت هر خوشه مشخص می‌شود:

```python
class clustering():
    def __init__(self, bayesian_network, attributes_domain):
        self.BN = bayesian_network
        self.attributes_domain = attributes_domain
        self.clusters = self.Attribute_Clustering()
        self.PBC = self.privacy_budget_coefficient()

```

### خوشه‌بندی
در تابع Attribute_Clustering متغیر S لیستی از ویژگی‌هایی است که در هیچ خوشه‌ای قرار نگرفته‌اند. متغیر clusters لیستی از خوشه‌ها است که در ابندا خالی است. برای انجام خوشه‌بندی ابتدا یک ویژگی تصادفی از S انتخاب می‌کنیم markov blanket  آن را که بعدا توضیح داده می‌شود محاسبه می‌کنیم. سپس ویژگی تصادفی انتخاب شده را درصورتی که در markov blanket(MB) محاسبه شده وجود نداشته باشد، با MB اضافه می‌کنیم و MB محاسبه شده را به عنوان یک خوشه درنظر گرفته و به clusters اضافه می‌کنیم. و درنهایت تمام ویژگی‌های موجود در MB را از S حذف می‌کنیم و این مراحل را تا جایی که در S متغیر وجود داشته باشد ادامه می‌دهیم:
```python
    def Attribute_Clustering(self):
        S = list(self.attributes_domain.keys())
        
        clusters = []  # clusters

        while len(S) > 0:
            random_attribute = random.choice(S)
            MB = self.markove_blanket(random_attribute)

            if random_attribute not in MB:
                MB.insert(0, random_attribute)
                
            clusters.append(MB)

            [S.remove(x) for x in MB if x in S]

        return clusters
```
#### markov blanket(MB)
ورودی این تابع یک متغیر است و خروجی آن پدران و فرزندان متغیر موردنظر به همراه پدرانِ فرزندانِ متغیر موردنظر است(فرمول و کد در زیر قابل مشاهد است ).

پدران یک متغیر با فراخوانی تابع attribute_parents و فرزندان یک متغیر را با فراخانی تابع attribute_children محاسبه می‌شود.
```python
    # MB(x) = Pa(x) ⋃ Ch(x) ⋃ {Pa(y)∣y ∈ Ch(x)}
    # x = atr in my code
    def markove_blanket(self, atr):
        MB = []

        # Pa(x)
        parents = self.attribute_parents(atr)
        [MB.append(x) for x in parents if x not in MB]

        # Ch(x)
        children = self.attribute_children(atr)
        [MB.append(x) for x in children if x not in MB]

        for child in children:

            # {Pa(y)∣y ∈ Ch(x)}
            child_parents = self.attribute_parents(child)
            [MB.append(x) for x in child_parents if x not in MB]

        return MB
```

در تابع attribute_parents، پدران متغیری که به عنوان ورودی به این تابع داده می‌شود، مشخص می‌شود:

```python
    def attribute_parents(self, atr):
        parents = []
        for AP in self.BN:
            if AP[0] == atr:
                parents.extend(list(AP[1]))
        return parents
```

در تابع attribute_children، فرزندان متغیری که به عنوان ورودی به این تابع داده می‌شود، مشخص می‌شود:

```python
    def attribute_children(self, atr):
        children = []
        for AP in self.BN:
            if atr in AP[1]:
                children.extend([AP[0]])
        return children
    
```

با استفاده از توابع نام برده شده، عملیات خوشه‌بندی ویژگی‌ها انجام می‌گیرد، حال باید درجه اهمیت هر خوشه مشخص شود. برای اینکار از تابع privacy_budget_coefficient استفاده می‌شود

### درجه اهمیت هر خوشه

![photo](/img/PBC.png)

```python
    # PBC(CLi) = ( 1/CIF(CLi) )/( ∑ (j=1 to j=t )1/CIF(CLj) )
    # t = number of cluster
    def privacy_budget_coefficient(self):
        cl_cif = self.importance_factor()
        sum_of_reverse_cif = 0
        for cif in cl_cif:
            sum_of_reverse_cif += (1/cif)

        cl_pbc = []
        for cif in cl_cif:
            pbc = ((1/cif)/sum_of_reverse_cif)
            cl_pbc.append(pbc)
        
        return cl_pbc
```

در تابع قبلی از importance_factor(CIF) تابع این استفاده می‌شود:

![photo](/img/CIF.png)
```python
    # CIF(CLi) = (∑ (Aj∈CLi) H(Aj) )/(∑ (k=1 to k=d) H(Ak))
    # d = number of attributs
    def importance_factor(self):
        attributes_entropy, sum_of_entropy = self.entropy()
        CIF = []
        for cl in self.clusters:
            cl_entropy = 0
            for atr in cl:
                cl_entropy += attributes_entropy.get(atr)
            cl_cif = (cl_entropy/sum_of_entropy)
            CIF.append(cl_cif)
        return CIF
```
و در تابع importance_factor از تابع entropy(H) استفاده می‌شود:

![photo](/img/entropy.png)

```python
    # H(Ai) = − ∑ (x∈Ωi) Pr[Ai = x]logPr[Ai = x]
    def entropy(self):
        ent = dict()
        sum_of_entropy = 0
        for atr in self.attributes_domain.keys():
            pr = (1/(len(self.attributes_domain.get(atr))))
            H = (pr)*(math.log(pr))*(-1)
            sum_of_entropy += H
            ent.update({atr:H})
        return ent, sum_of_entropy
```

### main.py

### مرحله چهارم

بعد از خوشه‌بندی و مشخص شدن درجه اهمیت هر خوشه، این اطلاعات در اختیار کاربران قرار گرفته و حال آنها می‌توانند داده‌های خود را متناسب با درجه اهمیت نوفه‌دار کنند. به همین منظور، به ازای داده‌های هر کاربر second_perturbation فراخوانی می‌شود و خروجی آنها در یک لیست ذخیره می شود:

```python
        # client side
        # second perturbation
        clients2 = []
        for data in clients_dataset:
            rr2 = second_perturbation(e, clu.clusters, clu.PBC, f_dataset.domains, data)
            clients2.append(rr2.randomized_data)
               
        # convert list to dataframe
        clients2_df = pd.DataFrame(clients2, columns = f_dataset.dataset.columns.to_list())
```

بعد از اینکه همه کاربران داده های خود را نوفه‌دار کردند، لیست تولیدی به منظور استفاده در مرحله بعد به dataframe تبدیل می‌شود.

### second_perturbation

 در این کلاس از کتابخانه‌های زیر استفاده می‌شود:

```python
import numpy as np
import random
```
 
بعد از مقداردهی اولیه، تابع anonymizing فراخوانی می‌شود:

```python
class second_perturbation:
    def __init__(self, epsilon, clusters, PBC, attributes_domain, client_data):
        self.epsilon = epsilon
        self.clusters = clusters
        self.PBC = PBC
        self.attributes_domain = attributes_domain
        self.client_data = client_data
        self.randomized_data = self.anonymizing()
 ```

در تابع anonymizing به ازای هر ویژگی مشخص می‌شود که متعلق به کدام خوشه است(با فراخوانی cluster_detection). سپس بودجه حریم خصوصی‌ای که باید به خوشه پیدا شده، اختصاص یابد محاسبه می‌شود و بودجه محاسبه شده، بین ویژگی‌های موجود در آن خوشه تقسیم می‌شود. و درنهایت p محاسبه شده و به منظور تصادفی‌سازی مقدار ویژگی، تابع generalized_randomize_response فراخوانی شده و خروجی آن در لیست perturbation_attributes ذخیره می‌شود:

```python
    def anonymizing(self):
        perturbation_attributes = []
        for i in self.attributes_domain.keys():
            cluster_index = self.cluster_detection(i)
            e = self.epsilon * self.PBC[cluster_index]
            eps = e / len(self.clusters[cluster_index])
            p = np.exp(eps)/(np.exp(eps)+len(self.attributes_domain.get(i))-1)
            # q = (1-p)/(len(self.attributes_domain.get(i))-1)
            perturbation_attributes.append(self.generalized_randomize_response(self.client_data.get(i), self.attributes_domain.get(i), p))
        return perturbation_attributes
```

تابع cluster_detection برای تشخیص اینکه یک ویژگی متعلق به کدام خوشه است، به‌کارگرفته می‌شود:

```python 
    def cluster_detection(self, atr):
        for cl in self.clusters:
            if atr in cl:
                return self.clusters.index(cl)
        return -1
```

 ورودی تابع generalized_randomize_response شامل داده‌ای که قرار است تصادفی شود به همراه دامنه‌های ممکن و p است. در این تابع یک عدد تصادفی بین 0 و 1 انتخاب می‌شود و اگر کوچکتر از مقدار p باشد، داده تغییر نمی‌کند، در غیر این‌صورت از بین دامنه‌های این ویژگی یک مقدار دیگر مخالف با مقدار کنونی به صورت تصادفی انتخاب می‌شود: 

```python 
    def generalized_randomize_response(self, data, data_domain, p):
        coin = random.random()
        if coin <= p :
            return data
        else:
            return random.choice([ele for ele in data_domain if ele != data ])
           
```

### مرحله پنجم

در این مرحله آزمایشات total variation distance و mean squred error انجام می‌گیرد:

```python
        # total variation distance
        TVD(f_dataset.dataset, clients2_df, f_dataset.domains)
```

```python
        # Mean squared error
        MSE(f_dataset.dataset, clients2_df, f_dataset.domains)
        
```

### TVD.py

در این کلاس از کتابخانه‌ زیر استفاده شده است:

```python
import numpy as np
```

بعد از مقداردهی اولیه تابع total_variation_distance به منظور محاسبه TVD فراخوانی می‌شود:

```python
class TVD:
    def __init__(self, original_data, perturbed_data, domains):
        self.original_data = original_data
        self.perturbed_data = perturbed_data
        self.domains = domains
        self.total_variation_distance()

```

فرمول و کد TVD به شرح زیر می‌باشد:

![photo](/img/TVD.png)

where Ω is the domain of the probability variable X and Z; PX and PZ are the probability distributions of the original attribute variable X and the perturbed one Z.

```python
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
                atvd += abs(o_count - p_count)
            atvd = atvd * (1/2)*(1/len(self.original_data))
            tvd += atvd
        tvd = tvd * (1/len(self.domains.keys()))
        print("total variation distance = " + str(tvd))


```

از آنجایی که داده‌ ما ابعاد بالا است نتیجه را  تقسیم بر تعداد ویژگی‌ها کردیم. و چون می‌خواستیم نتیجه بین 0 و 1 باشد، آن را تقسیم بر اندازه مجموعه‌داده نیز کردیم.

### MSE.py

در این کلاس از کتابخانه‌ زیر استفاده شده است:

```python
import math
```

بعد از مقداردهی اولیه تابع meanـsquaredـerror به منظور محاسبه MSE فراخوانی می‌شود:

```python
class MSE:
    def __init__(self, original_data, perturbed_data, domains):
        self.original_data = original_data
        self.perturbed_data = perturbed_data
        self.domains = domains
        self.meanـsquaredـerror()
```

فرمول و کد MSE به شرح زیر می‌باشد:

![photo](/img/MSE.png)

```python
    def meanـsquaredـerror(self):
        mse = 0 
        for atr in self.original_data.columns:
            pmse = 0 # attribute total variation distance
            for d in self.domains.get(atr):
                o_count = 0
                p_count = 0
                for r in range(len(self.original_data)):
                    if str(self.original_data.at[r,atr]) == d:
                        o_count = o_count + 1
                    if str(self.perturbed_data.at[r,atr]) == d:
                        p_count = p_count + 1
                pmse += math.pow(abs(o_count - p_count), 2)
            pmse = pmse * (1/len(self.domains.get(atr)))
            mse += pmse
        mse = mse * (1/len(self.domains.keys()))*(1/len(self.original_data))
        print("Mean squared error = " + str(mse))
```

از آنجایی که داده‌ ما ابعاد بالا است نتیجه را  تقسیم بر تعداد ویژگی‌ها کردیم. و چون می‌خواستیم نتیجه بین 0 و 1 باشد، آن را تقسیم بر اندازه مجموعه‌داده نیز کردیم.

## نتایج
برای مجموعه داده با ۵۰ ویژگی(attributes_number = ۵۰) و ۱۰۰۰۰تا کاربر(record_number=۱۰۰۰۰):
```
epsilon = 1
total variation distance = 0.06174800000000001
Mean squared error = 155.74994034058594

epsilon = 5
total variation distance = 0.061768
Mean squared error = 155.59987985096075

epsilon = 10
total variation distance = 0.061796000000000045
Mean squared error = 155.9307110235325
```


برای مجموعه داده با ۵۰ ویژگی(attributes_number = ۵۰) و ۱۰۰۰تا کاربر(record_number=۱۰۰۰):

```
epsilon = 1
total variation distance = 0.1954200000000001
Mean squared error = 16.15927550195701

epsilon = 5
total variation distance = 0.19574000000000014
Mean squared error = 16.189203908716156

epsilon = 10
total variation distance = 0.19506000000000018
Mean squared error = 16.13715560631964
```


برای مجموعه داده با ۱۰۰ ویژگی(attributes_number = ۱۰۰) و ۱۰۰۰تا کاربر(record_number=۱۰۰۰):

```
epsilon = 1
total variation distance = 0.19771000000000027
Mean squared error = 15.878240591418487

epsilon = 5
total variation distance = 0.19794000000000012
Mean squared error = 15.913670698879201

epsilon = 10
total variation distance = 0.19771000000000002
Mean squared error = 15.928150099207397
```
