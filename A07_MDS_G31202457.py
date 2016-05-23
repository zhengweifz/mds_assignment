# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 17:05:08 2015

@author: vagrant
"""

import matplotlib.pyplot as plt  # 2D plotting
from sklearn import manifold  # multidimensional scaling
import pandas as pd
import argparse as ag
import scipy.spatial.distance as dist
from scipy.spatial.distance import squareform

def read_data(file_name):
    items = pd.read_excel(file_name)
    return items.values #return numpy ndarray 

def run_MDS(data):
    data = data.T #transpose to compare breakfast items
    distance_vec = dist.pdist(data,"euclidean")
    distance_matrix = squareform(distance_vec)#covert to matrix form
    mds_method = manifold.MDS(n_components = 2, random_state = 9999,\
    dissimilarity = 'precomputed')
    mds_method.fit(distance_matrix)
    mds_coordinates = mds_method.fit_transform(distance_matrix)                                                                                                                                  
    labels = ['toast pop-up', 'buttered toast', 'English muffin and margarine', 'jelly donut', 'cinnamon toast',
                  'blueberry muffin and margarine', 'hard rolls and butter', 'toast and marmalade',
                  'buttered toast and jelly', 'toast and margarine', 'cinnamon bun','Danish pastry','glazed donut','coffee cake','corn muffin and butter']    
    # plot mds solution in two dimensions using item labels
    # defined by multidimensional scaling
    plt.figure()
    plt.scatter(mds_coordinates[:,0],mds_coordinates[:,1],\
         facecolors = 'none', edgecolors = 'none')  # points in white (invisible)
    for label, x, y in zip(labels, mds_coordinates[:,0], mds_coordinates[:,1]):
        plt.annotate(label, (x,y), xycoords = 'data')
    plt.xlabel('First Dimension')
    plt.ylabel('Second Dimension')    
    plt.show()


def main():
    parser = ag.ArgumentParser()
    parser.add_argument("file_name", help="Please enter input file name!")
    myArgs = parser.parse_args()
    data = read_data(myArgs.file_name)
    run_MDS(data)
    
if __name__ == "__main__":
    main()


