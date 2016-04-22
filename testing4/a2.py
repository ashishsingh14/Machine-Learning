import sys, os
from sklearn import cluster
import numpy as np
from matplotlib import pyplot
import matplotlib.dates as mdates
from datetime import datetime
from math import sqrt
import pandas as pd
from operator import itemgetter

def plotrelationships():
    district = np.genfromtxt("Crimes1.csv", dtype=None, delimiter=',', unpack=True, usecols=(11), skip_header=1)
    ward = np.genfromtxt("Crimes1.csv", dtype=None, delimiter=',', unpack=True, usecols=(12), skip_header=1)
    community = np.genfromtxt("Crimes1.csv", dtype=None, delimiter=',', unpack=True, usecols=(13), skip_header=1)
    crime = np.genfromtxt("Crimes1.csv", dtype=None, delimiter=',', unpack=True, usecols=(14), skip_header=1)
    
    #data = np.transpose(data)
    bad = []
    for i in range(80000):
        if district[i].isdigit()==False or ward[i].isdigit()==False or community[i].isdigit()==False:
            bad.append(i)
    district = district.tolist()
    ward = ward.tolist()
    community = community.tolist()
    crime = crime.tolist()
        
    del district[80000:]
    del ward[80000:]
    del community[80000:]
    del crime[80000:]  
    print len(bad)
    
    for i in range(80000):
        if crime[i]=='01A':
            crime[i] = int('1')
        elif crime[i]=='04A':
            crime[i] = int('4')
        elif crime[i]=='04B':
            crime[i] = int('41')
        elif crime[i]=='08A':
            crime[i] = int('8')
        elif crime[i]=='08B':
            crime[i] = int('81')
        else:
            crime[i] = int(crime[i])
    print crime[0:112]

    for i in range(80000):
        if i not in bad:
            district[i] = int(district[i])
            ward[i] = int(ward[i])
            community[i] = int(community[i])
            crime[i] = int (crime[i])
    
    
    ward1 = []
    crime1 = []
    community1 = []
    district1 = []
    print len(crime)
    print len(ward)
    print len(community)
    print len(district)
    
    for i in range(80000):
        if i not in bad:
            ward1.append(ward[i])
            crime1.append(crime[i])
            community1.append(community[i])
            district1.append(district[i])
            
    
    
    
    data1= []
    data1.append(crime1)    
    data1.append(district1)
    data1.append(community1)
    data1.append(ward1)
    
    data = np.asarray(data1)
    data = np.transpose(data)
    return data
    #print data
    

def kmeans(data, clusters):                #7, is good   
    kmeans = cluster.KMeans(n_clusters= clusters)
    kmeans.fit(data)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    print ("Number of unique clusters are: %d", clusters)
    #print labels
    
    points_per_cluster = [0 for x in range(clusters)]
    for i in xrange(len(data)):
        points_per_cluster[labels[i]] = points_per_cluster[labels[i]] + 1
    
    """
    mx = 9999999
    index1 = -1
    
    mn = -9999999
    index2 = -1
    for i in range(len(points_per_cluster)):
        if points_per_cluster[i] < mx:
            mx = points_per_cluster[i]
            index1 = i
        elif points_per_cluster[i] > mn:
            mn = points_per_cluster[i]
            index2 = i
    
    print "\nCluster Showing Anomalies:\n"
    
    for i in xrange(len(data)):
        if (labels[i]==index1):
            print data[i]
            
    print "\nNormal Cluster:\n"   
    for i in xrange(len(data)):
        if (labels[i]==index2):
            print data[i]"""

    print points_per_cluster
    return points_per_cluster
    


if __name__=="__main__":
    data = plotrelationships()
    kmeans(data, 15)
    






