#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 10:36:42 2019

@author: yajunbai
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import math




# problem a
# Parameters:
# X - is the input data
# ic - is the initial center
# iters - is the maximum iterations
# plot_progress - if True plots a figure of the data after each iteration
def Kmeans(X,centroids,iters=1,plot_progress=None):
    m,n=X.shape
    K=centroids.shape[0]
    previous_centroids=np.zeros((iters, centroids.shape[0], centroids.shape[1]))
    idx=np.zeros(m)
    for i in range (iters):
        previous_centroids[i,:]=centroids
        idx=find_closest_center(X,centroids)
        centroids = computecentroids(X, idx, K)
        if plot_progress:
            plotdatapts(X, idx, K, f'K={len(centroids)}', i=i+1)
    return centroids, idx

# Compute the closest center to the datapoints
def find_closest_center(X, center):
    m=X.shape[0]
    idx=np.zeros(m)
    for i in range (m):
        dist=np.square(np.sum(abs(X[i,:]-center)**2,axis=1))
        idx[i]=np.argmin(dist)           
    return idx
# plots 2d datapoints
def plotdatapts(X,idx,K,t,centers=np.zeros((1,1)), i=None):
    if i == None:
        plt.title(t)
    else:
        plt.title(f'{t} interation={i}')
    color=cm.rainbow(np.linspace(0,1,K))
    plt.scatter(X[:,0],X[:,1],c=color[idx.astype(int),:])
    if centers.shape[1] > 1:
        plt.scatter(centers[:,0],centers[:,1],c='g', marker='+')
    
# plots 3d datapoints
def plotdatapts3d(X,idx,K,i=None):
    fig = plt.figure(figsize=(6,4))
    ax = Axes3D(fig)
    if i != None:
        plt.title(i)
    color=cm.rainbow(np.linspace(0,1,K))
    ax.scatter(X[:,0],X[:,1],X[:,2],c=color[idx.astype(int),:], marker='o')

# Compute the center by computing the mean of every cluster as a new center
def computecentroids(X,idx,K):
    m,n=X.shape
    centroids=np.zeros((K,n))
    for i in range (K):
        x=X[idx==i]
        if x.shape[0]>0:
            avg=np.mean(x,axis=0)
            centroids[i,:]=avg
    return centroids

# Compute the k-means cost for the clustering induced by given centers
def calculate_cost(X, idx, centroids):
    cost = 0
    for i in range(math.floor(idx.max())+1):
        datapoints = np.where(idx == i)
        center = centroids[i]
        for j in range(len(datapoints[0])):
            cost+=np.square(np.sum(abs(X[datapoints[0][j]] - center)**2))
    return cost
