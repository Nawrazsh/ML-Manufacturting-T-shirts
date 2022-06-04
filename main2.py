import pandas as pd
from pca import PCA
import k_mean as km
import plot as pl
import time as t
n=100000
k=input()
data=pd.read_csv('group5.csv',names=["column 1","column 2","column 3","column 4","column 5","column 6","pca 1","pca 2","column 7"])
start=t.time()
reduced_data=PCA(n_components=2,data=data)
data["pca 1"]=reduced_data[:,0]
data["pca 2"]=reduced_data[:,1]
data,Centers=km.k_mean_pca(data=data,k=k)
end=t.time()
print("total runtime is ",end-start," for 100000 sample")
print(data.groupby(["column 7"]).size()/n*100)
pl.plot(k,data,Centers)
