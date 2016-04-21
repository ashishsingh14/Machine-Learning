import sys, os
from sklearn import cluster
import numpy as np
from matplotlib import pyplot
import matplotlib.dates as mdates
from datetime import datetime
from math import sqrt
import pandas as pd
from operator import itemgetter

def formatcsvfile():
    #converters={0:mdates.strpdate2num('%b-%d-%Y')}
    days = np.genfromtxt("sample1.csv", dtype=None, delimiter=',', unpack=True, usecols=(0,), skip_header=1)
    milvisits = np.genfromtxt("sample1.csv", dtype=None, delimiter=',', unpack=True, usecols=(13,), skip_header=1)
    modified_days = []
    for element in days:
        y = element.strip('"')
        modified_days.append(datetime.strptime(y, "%b-%d-%Y"))
    modified_nd = np.asarray(modified_days)
    
    #print modified_nd
    #print milvisits
    pyplot.scatter(modified_nd, milvisits)
    pyplot.title("Visits Vs Time")
    pyplot.ylabel("Visits")
    pyplot.xlabel("Time")
    pyplot.grid(True)
    pyplot.show()
    #print modified_nd
    
def generatedata():
    days = np.genfromtxt("sample1.csv", dtype=None, delimiter=',', unpack=True, usecols=(0,), skip_header=1)
    milvisits = np.genfromtxt("sample1.csv", dtype=None, delimiter=',', unpack=True, usecols=(13,), skip_header=1)
    modified_days = []
    for element in days:
        y = element.strip('"')
        modified_days.append(datetime.strptime(y, "%b-%d-%Y"))
    modified_nd = np.asarray(modified_days)
    
    #print modified_days
    for i in range(len(modified_days)):
        modified_days[i] = (int)(modified_days[i].strftime('%s'))
    modified_nd = np.asarray(modified_days)
    #print type(modified_days[0])
    
    #print modified_nd
    data = np.column_stack((modified_nd, milvisits))
    return data

"""
def examineclusters(clusters):
    results = pd.DataFrame({ 'cluster' : clusters, 'class' : kdd_data['class']})
    cluster_counts = results.groupby('cluster')['class'].value_counts()
    for i in xrange(len(cluster_counts)):
        print("Cluster " ,i)
        print(cluster_counts[i])"""

def kmeans(data):
    """
    days = np.genfromtxt("sample1.csv", dtype=None, delimiter=',', unpack=True, usecols=(0,), skip_header=1)
    milvisits = np.genfromtxt("sample1.csv", dtype=None, delimiter=',', unpack=True, usecols=(13,), skip_header=1)
    modified_days = []
    for element in days:
        y = element.strip('"')
        modified_days.append(datetime.strptime(y, "%b-%d-%Y"))
    modified_nd = np.asarray(modified_days)
    
    #print modified_days
    for i in range(len(modified_days)):
        modified_days[i] = (int)(modified_days[i].strftime('%s'))
    modified_nd = np.asarray(modified_days)
    #print type(modified_days[0])"""
    
    #print modified_nd
    #data = generatedata()
    kmeans = cluster.KMeans(n_clusters= 8)
    kmeans.fit(data)
    clusters = kmeans.predict(data)
    #print "clusters ", clusters
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    colors = ["g.","r.","c.","y.","b.","m.","k.","w."]
    #print labels
    #print centroids
    for i in range(len(data)):
        #print ("coordinates :", data[i], "label:" , labels[i])
        pyplot.plot(data[i][0], data[i][1],colors[labels[i]], markersize = 10)
    pyplot.scatter(centroids[:, 0],centroids[:, 1], marker = "x", s=150, linewidths = 5, zorder = 10)
    pyplot.show()


def dbscan():
    data = generatedata()
    db = cluster.DBSCAN(eps=1.0, min_samples=15).fit(data)
    labels = db.labels_
    unique_labels = np.unique(labels)
    n_clusters_ = len(unique_labels)
    
    print labels
    """
    core_samples = np.zeros_like(labels, dtype = bool)
    core_samples[db.core_sample_indices_] = True
    unique_labels = np.unique(labels)
    #colors = pyplot.cm.Spectral(np.linspace(0,1, len(unique_labels)))
    print('Estimated number of clusters: %d' % n_clusters_)
    
    unique_labels = set(labels)
    colors = 10*["g.","r.","c.","y.","b.","m.","k.","w."]
    
    for (label, color) in zip(unique_labels, colors):
        class_member_mask = (labels == label)
        xy = data[class_member_mask & core_samples]
        pyplot.plot(xy[:,0],xy[:,1], 'o', markerfacecolor = color, markersize = 10)
        
        xy2 = data[class_member_mask & ~core_samples]
        pyplot.plot(xy2[:,0],xy2[:,1], 'o', markerfacecolor = color, markersize = 5)
        
    pyplot.show()"""


def calculateeuclideandistance(x1,y1,x2,y2):
    distance = sqrt( (x1-x2)**2 + (y1-y2)**2)
    return distance
        
def calculatedistance(centers, data, labels):
    distance = []
    for i in xrange(len(data)):
        d = calculateeuclideandistance(data[i][0], data[i][1], centers[labels[i]][0], centers[labels[i]][1])
        distance.append(d)
    #calculatethresholdvalue1(centers, data, labels, distance)
    return distance
        
def testingdata():
    y = [2,2,2,18,18,18]
    x = [1,1.5,2,6,7,8]
    data = np.column_stack((np.asarray(x), np.asarray(y)))
    return data

def calculatethresholdvalue1(data, labels, distance):
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    total_points = len(data)
    points_per_cluster = [0 for x in range(n_clusters_)]
    distance_per_cluster = [0 for x in range(n_clusters_)]
    for i in xrange(len(data)):
        points_per_cluster[labels[i]] = points_per_cluster[labels[i]] + 1
        distance_per_cluster[labels[i]] = distance_per_cluster[labels[i]] + distance[i]
    #print total_points
    #print distance_per_cluster
    threshold = 0
    for i in xrange(n_clusters_):
        threshold = threshold + distance_per_cluster[i]*points_per_cluster[i]
    threshold = float(threshold/total_points)
    return threshold

def calculatenearestpoint(data, labels, distance):
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    nearest_point_per_cluster = [9999999999 for x in range(n_clusters_)]
    for i in xrange(len(data)):
        nearest_point_per_cluster[labels[i]] = min(distance[i], nearest_point_per_cluster[labels[i]])
    
    return nearest_point_per_cluster

def markoutliers2(data, labels, distance,alpha):
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    total_points = len(data)    
    outliers = [0 for x in range(total_points)]
    nearest_point_per_cluster = calculatenearestpoint(data, labels, distance)
    for i in xrange(len(data)):
        if distance[i]==0:
            continue
        else:
            if float(nearest_point_per_cluster[labels[i]]/distance[i]) > alpha:
                outliers[i] = 1
    return outliers

def markoutliers3(data, labels, distance, min_points):
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    total_points = len(data)
    points_per_cluster = [0 for x in range(n_clusters_)]
    for i in xrange(len(data)):
        points_per_cluster[labels[i]] = points_per_cluster[labels[i]] + 1
    outliers = [0 for x in range(total_points)]
    
    for i in xrange(n_clusters_):
        if points_per_cluster[i]<min_points:
                continue
        else:
            each_cluster = []
            for j in xrange(total_points):
                temp = []
                if labels[j]==i:
                    temp.append(distance[j])
                    temp.append(j)
                    each_cluster.append(temp)
            sorted_per_cluster = sorted(each_cluster, key=itemgetter(0),reverse=True)
            print sorted_per_cluster
            for l in xrange(min_points):
                outliers[sorted_per_cluster[l][1]] = 1
    return outliers


def markoutliers1(data, labels, distance,threshold):
    #second method
    total_points = len(data)    
    outliers = [0 for x in range(total_points)]
    for i in xrange(len(data)):
        if (distance[i]>threshold):
            outliers[i] = 1
    return outliers
       
def meanshift(data):
    ms = cluster.MeanShift()
    ms.fit(data)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print ("Number of unique clusters are: %d", n_clusters_)
    colors = 10*["g.","r.","c.","y.","b.","m.","w."]
    print labels
    
    for i in xrange(len(cluster_centers)):
        print cluster_centers[i][0], cluster_centers[i][1]
        print "\n"
    
    distance = calculatedistance(cluster_centers, data, labels)
    print distance
    #threshold = calculatethresholdvalue1(data, labels, distance)
    #print threshold
    """
    alpha = 0.0001
    while alpha<0.1:
        outliers = markoutliers2(data, labels, distance, alpha)
        alpha = alpha + 0.0002
        print outliers"""
    outliers = markoutliers3(data, labels, distance, 3)
    print "outliers: ", outliers
    for i in range(len(data)):
        #print ("coordinates :", data[i], "label:" , labels[i])
        if outliers[i]==1:
            pyplot.plot(data[i][0], data[i][1],"k.", markersize = 20)
        else :
            pyplot.plot(data[i][0], data[i][1],colors[labels[i]], markersize = 10)
    pyplot.scatter(cluster_centers[:, 0],cluster_centers[:, 1], marker = "x", s=150, linewidths = 5, zorder = 10)
    pyplot.show()
    pyplot.clf()

    
if __name__ == "__main__":
    data = testingdata()
    meanshift(data)
    """
    a = [[2,4],[5,8],[999,1]]
    b = sorted(a, key=itemgetter(0), reverse=True)
    print b"""
    
    
    
    
