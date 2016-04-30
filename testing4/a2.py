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
from a3 import plotyearwisecluster

years = ['2001', '2002', '2003','2004', '2005', '2006'] # ,'2004','2005','2006'
crimes = ["THEFT", "MOTOR VEHICLE THEFT", "BURGLARY", "ROBBERY", "ASSAULT", "ARSON",
           "BATTERY", "CRIM SEXUAL ASSAULT", "CRIMINAL DAMAGE", "CRIMINAL TRESPASS", "DECEPTIVE PRACTICE", "GAMBLING",
            "HOMICIDE", "INTERFERENCE WITH PUBLIC OFFICER", "INTIMIDATION", "KIDNAPPING", "LIQUOR LAW VIOLATION", 
            "NARCOTICS", "OBSCENITY", "OFFENCE INVOLVING CHILDREN", "OTHER NARCOTIC VIOLATION", "OTHER OFFENCE", 
            "PROSTITUTION", "PUBLIC INDECENCY", "PUBLIC PEACE VIOLATION", "RITUALISM", "SEX OFFENCE", "STALKING", 
            "WEAPONS VIOLATION"]

code_crime1 = ['2','3','5','6','7','9','10','11','12','13','14','15','16','17','18','19','20',
              '22','24','26','01A','04A','04B','08A','08B']

code_crime = ['6', '7', '5', '3', '08A', '9', '08B', '2', '14', '26', '10', '19', '01A', '24', '26', '26', '22', '18', '17', '26', '18', '26', '16', '17', '26', '04B', '17', '08A', '15']

def yearwiseresults():
    f = open("Crimes1.csv", "r")
    write_results = open("results.txt", "wa")
    for y in xrange(len(years)):
        date = []
        crime = []
        freq_crime = [0 for x in range(len(crimes))]
        for c in xrange(len(code_crime)):
            for line in f:
                a = line.split(",")
                #print a
                if  a[14]==code_crime[c] and a[17]== years[y]:
                    #print "hello"
                    date.append(a[2])
                    crime.append(c)
                    freq_crime[c] = freq_crime[c] + 1
            f.seek(0)
        modified_date = []
        image_name = "crime year " + years[y]
        print freq_crime
        total_crimes = sum(freq_crime)
        write_results.write("\n")
        write_results.write(years[y])
        write_results.write("\n")
        for i in xrange(len(freq_crime)):
            write_results.write(crimes[i])
            write_results.write("\t")
            p1 = float(freq_crime[i])/float(total_crimes)
            perc = p1*100.0
            #print perc
            st = str(perc)
            print st
            write_results.write(st)
            write_results.write("\n")
        write_results.write("\n")
        """
        for element in date:
            y = element.split(" ")
            day = y[0]
            modified_date.append(datetime.strptime(day, "%m/%d/%Y"))
        
        pyplot.scatter(modified_date, crime)
        #pyplot.show()
        pyplot.savefig(image_name, bbox_inches='tight')
        pyplot.clf()
        print modified_date
        print crime"""
        #print modified_date
    

def yearwiseclustering():
    f = open("Crimes1.csv", "r")
    for y in xrange(len(years)):
        print "For year: ", years[y]
        crime = []
        district = []
        community_area = []
        for line in f:
            a = line.split(",")
            if a[17] == years[y] and a[11].isdigit()==True and a[13].isdigit()==True:
                cd = a[14]
                if cd=='01A':
                    crime.append(int('1'))
                elif cd=='04A':
                    crime.append(int('4'))
                elif cd=='04B':
                    crime.append(int('41'))
                elif cd=='08A':
                    crime.append(int('8'))
                elif cd=='08B':
                    crime.append(int('81'))  
                else:
                    crime.append(int(cd))
                district.append(int(a[11]))
                community_area.append(int(a[13]))
                
        f.seek(0)
        data1= []
        data1.append(crime)    
        data1.append(district)
        data1.append(community_area)
        data = np.asarray(data1)
        data = np.transpose(data)
        kmeans(data, 5)
        

def generatedata():
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
    #data1.append(community1)
    #data1.append(ward1)
    
    data = np.asarray(data1)
    data = np.transpose(data)
    return data
    #print data
    

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
    centroids = ms.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print ("Number of unique clusters are: %d", n_clusters_)
    
    points_per_cluster = [0 for x in range(n_clusters_)]
    for i in xrange(len(data)):
        points_per_cluster[labels[i]] = points_per_cluster[labels[i]] + 1
    
    print "Points per cluster\n"
    print points_per_cluster
    
    
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
    
    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = ["g","r","c","y","b","m","w"]
    for i in range(1000):
        ax.scatter(data[i][0], data[i][1], data[i][2], zdir='z', c = colors[labels[i]])
    ax.scatter(centroids[:, 0],centroids[:, 1], centroids[:, 2], zdir='z', marker = "x", s=200, linewidths = 5, zorder = 10)
    ax.set_xlabel('Crime')
    ax.set_ylabel('District')
    ax.set_zlabel('Community')
    
    pyplot.show()
    
    print "\nCluster Showing Anomalies:\n"
    
    for i in xrange(len(data)):
        if (labels[i]==index1):
            print data[i]

    return points_per_cluster
    


if __name__=="__main__":
    
    #yearwiseclustering()
    yearwiseresults()
    """
    data = generatedata()   
    points_per_cluster = kmeans(data, 6)
    print points_per_cluster
    
    
    clusters = []
    variances = []
    for i in range(5,30):                        # cluster = 5 is good for resp
        points_per_cluster = kmeans(data, i)
        variances.append(np.var(points_per_cluster))    
        clusters.append(i)
    
    print "data obtained"
    print clusters
    print variances
    pyplot.plot(clusters, variances)
    pyplot.title("Showing Variation of Variance with Total Clusters")
    pyplot.xlabel("No of Clusters -->")
    pyplot.ylabel("Variance -->")
    pyplot.show()
    
    plotyearwise()"""
    
    






