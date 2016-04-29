import os, sys
from sklearn import cluster
import numpy as np
from matplotlib import pyplot
import matplotlib.dates as mdates
from datetime import datetime
from math import sqrt
import pandas as pd
from operator import itemgetter
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import linkage, dendrogram
import scipy.spatial.distance as ssd
from numpy import array
from statsmodels.stats.moment_helpers import mnc2cum

def plottrendanalysis(filename):
    with open(filename, 'r') as f:
        first = f.readline()
        f.readline()
        third = f.readline()
        first1 = first.split()
        diseases_name = first1[1]
        #print diseases_name
        third = third.split(",")
        third = third[2].strip("\n")
        year_week = []
        cases = []
        flag = 0
        for line in f:
            a = line.split(',')
            b = float(a[0] + '.' + a[1])
            year_week.append(b)
            afterstrip = a[2].strip("\n")
            if afterstrip.isdigit():
                cases.append(float(a[2]))
            else:
                cases.append(0.0)
        image_name = first + third
        image_name1 = "Trend-" + diseases_name + "-" + third + ".png"
    #print len(year_week)
    #print len(cases)
    pyplot.plot(year_week, cases)
    #pyplot.xlim(1965, 2003)
    pyplot.title(image_name)
    pyplot.ylabel("Cases corresponding week")
    pyplot.xlabel("Year")
    pyplot.grid(True)
    pyplot.savefig(image_name1, bbox_inches='tight')
    pyplot.clf()
    f.close()


def generatedata():
    year = np.genfromtxt("HEPATITIS_A_Cases_1966-2012_20160414022313.csv", dtype=None, delimiter=',', unpack=True, usecols=(0), skip_header=3)
    week = np.genfromtxt("HEPATITIS_A_Cases_1966-2012_20160414022313.csv", dtype=None, delimiter=',', unpack=True, usecols=(1), skip_header=3)
    cases = np.genfromtxt("HEPATITIS_A_Cases_1966-2012_20160414022313.csv", dtype=None, delimiter=',', unpack=True, usecols=(2), skip_header=3)
    #data  = np.genfromtxt("HEPATITIS_A_Cases_1966-2012_20160414022656.csv", dtype=None, delimiter=',', unpack=True, usecols=(0,1,2), skip_header=3)
    new_cases = []
    
    for i in xrange(len(cases)):
        if cases[i].isdigit():
            new_cases.append(int(cases[i]))
        else:
            new_cases.append(0)
    
    data1 = []
    data1.append(year)
    data1.append(week)
    data1.append(new_cases)
    data = np.asarray(data1)
    print (type(data))
    data = np.transpose(data)
    #print data
    return data
    #print new_cases[-52:]
    
    
def kmeans(data, clusters):
    """
    kmeans = cluster.KMeans(n_clusters= clusters)
    kmeans.fit(data)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    print ("Number of unique clusters are: %d", clusters)
    #print labels"""
    
    ms = cluster.MeanShift()
    ms.fit(data)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print ("Number of unique clusters are: %d", n_clusters_)
    
    points_per_cluster = [0 for x in range(n_clusters_)]
    for i in xrange(len(data)):
        points_per_cluster[labels[i]] = points_per_cluster[labels[i]] + 1
    
    print "Points per cluster\n"
    print points_per_cluster
    
    """
    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = ["g","r","c","y","b","m","w"]
    for i in range(len(data)):
        ax.scatter(data[i][0], data[i][1], data[i][2], zdir='z', c = colors[labels[i]])
        
    ax.scatter(centroids[:, 0],centroids[:, 1], centroids[:, 2], zdir='z', marker = "x", s=200, linewidths = 5, zorder = 10)"""
    
    print "Points per cliuster\n"
    
    for i in xrange(clusters):
        print "Cluster: ", i
        print "\n"
        for j in xrange(len(data)):
            if (labels[j]==i):
                print data[j]
    
    #20 - 32 summer 
    #45 - 10 winter

def euclideanfornpoints(a,b):
    length = len(a)
    sum = 0
    for i in xrange(length):
        sum = sum + (a[i]-b[i])**2
    sum = sqrt(sum)
    return sum
  

def plotyearwisecluster(filename):
    year = np.genfromtxt(filename, dtype=None, delimiter=',', unpack=True, usecols=(0), skip_header=3)
    week = np.genfromtxt(filename, dtype=None, delimiter=',', unpack=True, usecols=(1), skip_header=3)
    cases = np.genfromtxt(filename, dtype=None, delimiter=',', unpack=True, usecols=(2), skip_header=3)
    #data  = np.genfromtxt("HEPATITIS_A_Cases_1966-2012_20160414022656.csv", dtype=None, delimiter=',', unpack=True, usecols=(0,1,2), skip_header=3)
    f = open(filename, "r")
    first = f.readline()
    f.readline()
    third = f.readline()
    first1 = first.split()
    diseases_name = first1[1]
    #print diseases_name
    third = third.split(",")
    third = third[2].strip("\n")
    #print third
    new_cases = []
    
    for i in xrange(len(cases)):
        if cases[i].isdigit():
            new_cases.append(int(cases[i]))
        else:
            new_cases.append(0)
    
    year = year.tolist()
    week = week.tolist()
    unique_years = set(year)
    unique_years  = list(unique_years)
    
    
    weekly_per_year = []
    
    total_years = len(unique_years)
    index = 0
    for i in xrange(total_years):
        temp1 = new_cases[index : index+52]
        weekly_per_year.append(temp1)
        index = index + 52
    
    distance_matrix = []
    for i in xrange(len(weekly_per_year)):
        temp = []
        for j in xrange(len(weekly_per_year)):
            if i==j:
                temp.append(0)
            else:
                distance = euclideanfornpoints(weekly_per_year[i], weekly_per_year[j])
                temp.append(distance)
        distance_matrix.append(temp)
    
    distArray = ssd.squareform(distance_matrix)
    
    linkage_matrix = linkage(distArray, 'single')
    #print linkage_matrix
    pyplot.figure(101)
    #pyplot.subplot(1, 2, 1)
    image_name = first + third
    pyplot.title(image_name)
    ar = []
    mn = min(unique_years)
    mx = max(unique_years) + 1
    for i in xrange(mn, mx, 1):
        ar.append(i)
    image_name1 = diseases_name + "-" + third + ".png"
    
    print image_name1
    dendrogram(linkage_matrix,color_threshold=1,truncate_mode='lastp',labels=ar,distance_sort='ascending')
    pyplot.savefig(image_name1, bbox_inches='tight')
    pyplot.clf()
    f.close()
    
    """
    data = []
    for i in xrange(total_years):
        for j in xrange(i+1,total_years,1):
            temp = []
            temp.append(distance_matrix[i][j])
            temp.append(unique_years[i])
            temp.append(unique_years[j])
            data.append(temp)"""


def listallfiles():
    #source = '/home/ashish/Documents/Machine Learning/testing4/datadiseases'
    source = "/home/ashish/Documents/Machine Learning/testing4/california"
    for root, dirs, filenames in os.walk(source):
        for f in filenames:
            #print f
            fullpath = os.path.join(source, f)
            #print fullpath
            plottrendanalysis(fullpath)
            #plotyearwisecluster(fullpath)
            
            
if __name__=="__main__":
    
    #data = generatedata()
    #kmeans(data, 5)
    #plotyearwisecluster("/home/ashish/Documents/Machine Learning/testing4/HEPATITIS_A_Cases_1966-2012_20160414022313.csv")
    
    #listallfiles()
    
    listallfiles()
    """
   
    *****
    b = [[0,4,5], [4,0,8],[5,8,0]]
    distArray = ssd.squareform(b)
    print distArray
    
    for i in xrange(2,8,1):
        for j in xrange(i+1,8,1):
            print i,j
            print "\n"
    
    val = 0. # this is the value where you want the data to appear on the y-axis.
    ar = [1,2,3,4,5] # just as an example array
    print ar
    pyplot.plot(ar, np.zeros_like(ar) + val, 'o')
    pyplot.show()"""




