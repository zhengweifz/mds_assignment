# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 11:02:15 2015

@author: vagrant
"""

import pandas as pd
import numpy as np
from scipy.cluster.vq import kmeans, vq
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse as ag
from sklearn import preprocessing, decomposition
import warnings

warnings.filterwarnings('ignore') # surpress numpy warning

def read_data(file_name):
    items = pd.read_excel(file_name)
    return items.values #return numpy ndarray   

def run_kmeans(data, n):
    #scale the data
    data = data.T
    scaled_data = preprocessing.scale(data)
    #PCA Reduce dimesnion
    pca = decomposition.PCA()
    pca.fit(scaled_data)
    pca.n_components = n
    reduced_data = pca.fit_transform(scaled_data)
     # computing K-Means with K = 2 (2 clusters)
    centroids,dist = kmeans(reduced_data, n)
    # assign each sample to a cluster
    idx,distance = vq(reduced_data, centroids)
    colors = cm.rainbow(np.linspace(0, 1, n))
    data_colors = colors[idx]
    plt.scatter(reduced_data[:,0], reduced_data[:,1],c=data_colors, s=30) 
    plt.scatter(centroids[:,0],centroids[:,1],c='green',s=30, marker='s')
    plt.show()
    
def main():
    parser = ag.ArgumentParser()
    parser.add_argument("file_name", help="Please enter input file name!")
    parser.add_argument("-n", dest="n_c", action="store", help="number of cluster")
    myArgs = parser.parse_args()
    data = read_data(myArgs.file_name)
    run_kmeans(data, int(myArgs.n_c))
    
if __name__ == "__main__":
    main()