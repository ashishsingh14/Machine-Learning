import sys, os
from sklearn import cluster
import numpy as np
from matplotlib import pyplot
import matplotlib.dates as mdates
from datetime import datetime
from math import sqrt
import pandas as pd
from operator import itemgetter
from mpl_toolkits.mplot3d import Axes3D


def plotrelationships():
    days = np.genfromtxt("sample1.csv", dtype=None, delimiter=',', unpack=True, usecols=(0,), skip_header=1)
    milvisits = np.genfromtxt("sample1.csv", dtype=None, delimiter=',', unpack=True, usecols=(16,), skip_header=1)
    civvisits = np.genfromtxt("sample1.csv", dtype=None, delimiter=',', unpack=True, usecols=(18,), skip_header=1)
    prescriptions = np.genfromtxt("sample1.csv", dtype=None, delimiter=',', unpack=True, usecols=(17,), skip_header=1)
    dewpoint = np.genfromtxt("sample1.csv", dtype=None, delimiter=',', unpack=True, usecols=(23,), skip_header=1)
    wetbulb = np.genfromtxt("sample1.csv", dtype=None, delimiter=',', unpack=True, usecols=(24,), skip_header=1)
    avgtemp = np.genfromtxt("sample1.csv", dtype=None, delimiter=',', unpack=True, usecols=(21,), skip_header=1)
    isweekend = np.genfromtxt("sample1.csv", dtype=None, delimiter=',', unpack=True, usecols=(43,), skip_header=1)
    modified_days = []
    for element in days:
        y = element.strip('"')
        modified_days.append(datetime.strptime(y, "%b-%d-%Y"))
    modified_nd = np.asarray(modified_days)
    #print modified_nd
    # range 2-185 186-550 551-701  0-184 184:549 549:  [549:]
    
    pyplot.plot(modified_days[549:], milvisits[549:],'b')
    pyplot.plot(modified_days[549:], civvisits[549:], 'r')
    pyplot.plot(modified_days[549:], prescriptions[549:], 'g')
    pyplot.plot(modified_days[549:], avgtemp[549:] , 'k')
    pyplot.plot(modified_days[549:], wetbulb[549:], 'c')
    pyplot.plot(modified_days[549:], dewpoint[549:], 'm')
    
    pyplot.title("Showing Variations for Year 2003")
    pyplot.ylabel("Variation of different parameters --->")
    #pyplot.text("shosss")
    pyplot.xlabel("Time --->")
    pyplot.grid(True)
    pyplot.show()
    #pyplot.savefig("figure-2003", facecolor='w', edgecolor='w', transparent=False, pad_inches=0.5, bbox_inches='tight')
    #pyplot.clf()
 
   
def generatedata():
    #data = np.genfromtxt("sample1.csv", dtype=None, delimiter=',', unpack=True, usecols=(16,17,18,21,23,24,25), skip_header=1) #(13,14,15,21,23,24,35)
    days = np.genfromtxt("sample1.csv", dtype=None, delimiter=',', unpack=True, usecols=(0,), skip_header=1)
    milvisits = np.genfromtxt("sample1.csv", dtype=None, delimiter=',', unpack=True, usecols=(13,), skip_header=1)
    civvisits = np.genfromtxt("sample1.csv", dtype=None, delimiter=',', unpack=True, usecols=(15,), skip_header=1)
    prescriptions = np.genfromtxt("sample1.csv", dtype=None, delimiter=',', unpack=True, usecols=(14,), skip_header=1)
    dewpoint = np.genfromtxt("sample1.csv", dtype=None, delimiter=',', unpack=True, usecols=(23,), skip_header=1)
    wetbulb = np.genfromtxt("sample1.csv", dtype=None, delimiter=',', unpack=True, usecols=(24,), skip_header=1)
    avgtemp = np.genfromtxt("sample1.csv", dtype=None, delimiter=',', unpack=True, usecols=(21,), skip_header=1)
    isweekend = np.genfromtxt("sample1.csv", dtype=None, delimiter=',', unpack=True, usecols=(43,), skip_header=1)
    #sealevel = np.genfromtxt("sample1.csv", dtype=None, delimiter=',', unpack=True, usecols=(33,), skip_header=1)
    
    modified_days = []
    for element in days:
        y = element.strip('"')
        modified_days.append(datetime.strptime(y, "%b-%d-%Y"))
        
    for i in range(len(modified_days)):
        modified_days[i] = (int)(modified_days[i].strftime('%s'))
    data1 = []
    
    #data1.append(modified_days)
    data1.append(isweekend)
    data1.append(milvisits)
    data1.append(prescriptions)
    data1.append(civvisits)
    data1.append(avgtemp)
    data1.append(dewpoint)
    data1.append(wetbulb)
    #data1.append(sealevel)
    
    data = np.transpose(data1)
    print data[1]
    return data

def kmeans(data, clusters):                 #7, is good
    
    kmeans = cluster.KMeans(n_clusters= clusters)
    kmeans.fit(data)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    print ("Number of unique clusters are: %d", clusters)
    #print labels
    
    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d') 
    points_per_cluster = [0 for x in range(clusters)]
    for i in xrange(len(data)):
        points_per_cluster[labels[i]] = points_per_cluster[labels[i]] + 1
    
    mx = 9999999
    index1 = -1
    
    mn = -9999999
    index2 = -1
    print "Points per cluster\n"
    print points_per_cluster
    
    colors = ["g","r","c","y","b","m","w"]
    for i in range(len(points_per_cluster)):
        if points_per_cluster[i] < mx:
            mx = points_per_cluster[i]
            index1 = i
        elif points_per_cluster[i] > mn:
            mn = points_per_cluster[i]
            index2 = i
    
   
    for i in range(len(data)):
        if labels[i] == index1:
            ax.scatter(data[i][1], data[i][2], data[i][4], zdir='z', c = 'k')
        else:
            ax.scatter(data[i][1], data[i][2], data[i][4], zdir='z', c = colors[labels[i]])
        
    ax.scatter(centroids[:, 1],centroids[:, 2], centroids[:, 4], zdir='z', marker = "x", s=200, linewidths = 5, zorder = 10)
    ax.set_xlabel('Resp Visits')
    ax.set_ylabel('Prescriptions')
    ax.set_zlabel('Average Temperature')
    
    pyplot.show() 
    

    print "\nCluster Showing Anomalies:\n"
    
    for i in xrange(len(data)):
        if (labels[i]==index1):
            print data[i]
            
    print "\nNormal Cluster:\n"   
    for i in xrange(len(data)):
        if (labels[i]==index2):
            print data[i]

    return points_per_cluste
    

def meanshift(data):
    ms = cluster.MeanShift()
    ms.fit(data)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print ("Number of unique clusters are: %d", n_clusters_)
    print labels
    
    for i in xrange(len(cluster_centers)):
        print cluster_centers[i][0], cluster_centers[i][1]
        print "\n"


if __name__=="__main__":
    
    data = generatedata()
    kmeans(data, 7)
    #points_per_cluster = kmeans(data, 7)
    #kmeans(data,5)
    
    """
    clusters = []
    variances = []
    for i in range(3,21):                        # cluster = 5 is good for resp
        points_per_cluster = kmeans(data, i)
        variances.append(np.var(points_per_cluster))    
        clusters.append(i)
    
    print "data obtained"
    print clusters
    print variances
    variances[4] = 4012.2112
    pyplot.plot(clusters, variances)
    pyplot.title("Showing Variation of Variance with Total Clusters")
    pyplot.xlabel("No of Clusters -->")
    pyplot.ylabel("Variance -->")
    pyplot.show()
    
    plotrelationships()"""
    
    