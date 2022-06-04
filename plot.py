import matplotlib.pyplot as plt
import pandas as pd
color=["green","cyan","magenta","orange","black","purple","brown","beige"]
def plot(k,data,Centers):
    for i in range(k):
        tempData=data[data["column 7"]==i+1]
        plt.scatter(tempData["pca 1"],tempData["pca 2"],c=color[i])
        plt.scatter(Centers["pca 1"],Centers["pca 2"],c='red')
    plt.xlabel('pca 1')
    plt.ylabel('pca 2')
    plt.show()
    
