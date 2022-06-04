import pandas as pd
import scipy.stats as stats
from pca import PCA
import k_mean as km
import plot as pl
import time as t
import matplotlib.pyplot as plt
# the number of raws
n=100000
k=5 
# "The Height"=>"column 1"
# "The Weight"=>"column 2"
# "The body mass index"=>"column 3"
# "The length between the shoulders"=>"column 4"
# "The length of the arms"=>"column 5"
# "Cluster"=>"column 6"
# "pca 1"=>"pca 1"
# "pca 2"=>"pca 2"
# "Cluster"=>"column 7"
#Use Pandas to read csv into a list of lists with header
data=pd.read_csv('group5.csv',names=["column 1","column 2","column 3","column 4","column 5","column 6","pca 1","pca 2","column 7"])
#normalize the data
data=stats.zscore(data)
data["column 6"]=0
# k=eval(input("enter the number of clusters "))
# samples list
samples=[20000 ,30000 ,40000 ,50000 ,60000 ,70000 ,80000 ,90000,100000]
# time samples list
timeSmaples=[]
n_components = 2 
i=0
for sample in samples:
    # generate random samples
    # ValueError: Cannot take a larger sample than population when 'replace=False'
    tempData=data.iloc[0:sample,:]
    start=t.time()
    tempData=km.k_mean(data=tempData,k=k)
    end=t.time()
    timeSmaples.append(end-start)
    print(samples[i]," samples done ")
    i+=1
print(timeSmaples)
plt.plot(samples,timeSmaples)
plt.xlabel('numbr of samples')
plt.ylabel('running time')
plt.show()









