import math as m
import numpy as np
import random as random
K=0
# choose the cluster with the minimum distance 
def compareDistances(data):
    distances=[]
    for i in range(K):
        distances.append(data[f'dist{i+1}'])
    return np.argmin(distances)+1
# k-mean 5D => with complexity of O(k*n*number of iterations)
# for large number of n the complexity will be O(n)
def k_mean(data,k):
    global K
    K=k
    # Select k random points from the data as centroids to be the centers of clusters in first  iteration
    random.seed(1)
    # len(data.index)) number of rows
    randomlist = random.sample(range(0, len(data.index)), k)
    Centers=data.loc[randomlist]
    # change the row indexes to be the same as new centers list
    Centers.index=list(range(1,k+1))
    for i in range(k):
        data[f'dist{i+1}']=0
    while True:
        # calculate the difference between ceneters and each row(point)
        for i in range(k):
            data[f'dist{i+1}']=((data["column 1"]-Centers.at[i+1,"column 1"])**2+(data["column 2"]-Centers.at[i+1,"column 2"])**2+(data["column 3"]-Centers.at[i+1,"column 3"])**2+(data["column 4"]-Centers.at[i+1,"column 4"])**2+(data["column 5"]-Centers.at[i+1,"column 5"])**2)**0.5  
        # choose the cluster with the minimum distance 
        data["column 6"]=data.apply(compareDistances,axis=1)
        # calculate the new clusters centers by calculate the mean of each cluster
        newCenters = data.groupby(["column 6"]).mean()[["column 1","column 2","column 3","column 4","column 5"]]
        # calculate the differance between the old and the new centers
        dif=(newCenters["column 1"]-Centers["column 1"]).sum()+(newCenters["column 2"]-Centers["column 2"]).sum()+(newCenters["column 3"]-Centers["column 3"]).sum()+(newCenters["column 4"]-Centers["column 4"]).sum()+(newCenters["column 5"]-Centers["column 5"]).sum()
        if dif==0:
            break
        else:
            Centers=newCenters
        # print(dif)
        # print(Centers)





# k-mean 2D after the pca
def k_mean_pca(data,k):
    global K
    K=k
    # choose random points to be the centers of clusters in first  iteration
    random.seed(1)
    # len(data.index)) number of rows
    randomlist = random.sample(range(0, len(data.index)), k)
    Centers=data.loc[randomlist]
    # change the row indexes to be the same as new centers list
    Centers.index=list(range(1,k+1))
    while True:
        # calculate the difference between ceneters and each row
        for i in range(k):
            data[f'dist{i+1}']=((data["pca 1"]-Centers.at[i+1,"pca 1"])**2+(data["pca 2"]-Centers.at[i+1,"pca 2"])**2)**0.5  
            
        # choose the cluster with the minimum distance 
        data["column 7"]=data.apply(compareDistances,axis=1)
        # calculate the new clusters centers by calculate the mean of each cluster
        newCenters = data.groupby(["column 7"]).mean()[["pca 1","pca 2"]]
        # calculate the differance between the old and the new centers
        dif=(newCenters["pca 1"]-Centers["pca 1"]).sum()+(newCenters["pca 2"]-Centers["pca 2"]).sum()
        if dif==0:
            return data,Centers
        else:
            Centers=newCenters
        # print(dif)
        # print(Centers)
