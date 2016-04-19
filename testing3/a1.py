import sys, os
from sklearn import cluster
import numpy as np
from matplotlib import pyplot
import matplotlib.dates as mdates
from datetime import datetime

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
    

def kmeans():
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
    data = generatedata()
    kmeans = cluster.KMeans(n_clusters= 8)
    kmeans.fit(data)
    
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    colors = ["g.","r.","c.","y.","b.","m.","k.","w."]
    #print labels
    #print centroids
    for i in range(len(data)):
        print ("coordinates :", data[i], "label:" , labels[i])
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

def meanshift(data):
    ms = cluster.MeanShift()
    ms.fit(data)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print ("Number of unique clusters are: %d", n_clusters_)
    colors = 10*["g.","r.","c.","y.","b.","m.","k.","w."]
    print labels
    
    for i in range(len(data)):
        #print ("coordinates :", data[i], "label:" , labels[i])
        pyplot.plot(data[i][0], data[i][1],colors[labels[i]], markersize = 10)
    pyplot.scatter(cluster_centers[:, 0],cluster_centers[:, 1], marker = "x", s=150, linewidths = 5, zorder = 10)
    pyplot.show()
    pyplot.clf()
    

    
if __name__ == "__main__":
    data = generatedata()
    meanshift(data) 
    
    
    
    
