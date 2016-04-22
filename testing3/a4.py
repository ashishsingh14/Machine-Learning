import os, sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from a1 import meanshift
import numpy as np
from sklearn import cluster
states = ['Alabama', 'Alaska','Arizona','Arkansas','California','Colorado','Connecticut','Delaware','District of Columbia','Florida','Georgia',
              'Hawaii','Idaho','Illinois','Indiana','Iowa','Kansas','Kentucky','Louisiana','Maine','Maryland','Massachusetts','Michigan',
              'Minnesota','Mississipi','Missouri','Montana','Nebraska','Nevada','New Hampshire','New Jersey','New Mexico','New York','North Carolina'
              ,'North Dakota','Ohio','Oklahoma','Oregon','Pennsylvania','Rhode Island','South Carolina','South Dakota','Tennessee','Texas',
              'Utah','Vermont','Virginia','Washington','West Virginia','Wisconsin','Wyoming']
diseases = ["Alzheimer's disease",'Stroke','CLRD','Diabetes','Influenza and pneumonia','Cancer',"Parkinson's disease",
                'Pneumonitis due to solids and liquids','Septicemia']

diseases_not_coming = ['Homicide', 'Chronic liver disease and cirrhosis', 'Diseases of Heart', 
                       'Essential hypertension and hypertensive renal disease','Suicide','Kidney Disease']
years = [1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013]
num_diseases = [1,2,3,4,5,6,7,8,9]

def generatedata(a,b):
    modified_a = np.asarray(a)
    modified_b = np.asarray(b)
    data = np.column_stack((modified_a,modified_b))
    return data

def plotyearwise():
    f = open("NCHS_-_Age-adjusted_Death_Rates_for_the_Top_10_Leading_Causes_of_Death__United_States__2013.csv", "r")
    for i in range(1):
        for j in range(1):
            deaths = []
            for line in f:
                a = line.split(",")
                if a[0]==str(years[j]) and a[3]==states[i] and a[2]!='All Causes':
                    deaths.append(int(a[4]))
                    name = states[j]  + "-" + str(years[i])
                    image_name = name + ".png"                    
            plt.plot(num_diseases, deaths)
            plt.title(name)
            plt.savefig(image_name, bbox_inches='tight')
            plt.clf()
            print deaths
            data = generatedata(num_diseases, deaths)
            meanshift(data)
            f.seek(0)

def plotdiseasesforallyear():
    f = open("NCHS_-_Age-adjusted_Death_Rates_for_the_Top_10_Leading_Causes_of_Death__United_States__2013.csv", "r")
    for i in range(1):
        for j in range(1):
            year = []
            deaths = []
            for line in f:                
                a = line.split(",")
                if a[3]==states[i] and a[2]==diseases[j]:
                    year.append(int(a[0]))
                    deaths.append(int(a[4]))
                    name = states[i] + "-"+ diseases[j]
                    image_name = name + ".png"                    
            plt.plot(year, deaths)
            plt.title(name)
            plt.savefig(image_name, bbox_inches='tight')
            plt.clf()
            print year 
            print deaths
            data = generatedata(year, deaths)
            meanshift(data)
            #print year
            #print deaths
            f.seek(0)         

def plotin3d():   
    with open("NCHS_-_Age-adjusted_Death_Rates_for_the_Top_10_Leading_Causes_of_Death__United_States__2013.csv", "r") as f:
        f.readline()
        year = []
        deaths = []
        state_name = []
        cause_of_death = []
        for line in f:
            a = line.split(",")
            if a[3]=='Alabama' and a[2]!='All Causes':
                year.append(int(a[0]))
                deaths.append(int(a[4]))
                cause_of_death.append(a[2])
                state_name.append(a[3])
        print len(year)
        print len(cause_of_death)
        print len(deaths)
        
        print year
        print cause_of_death
        #plt.bar(year, deaths)
        #plt.show()
        
        #print deaths
        """
        z = 15 * [2,3,4,12,22,23,111,34,45,21]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(year, deaths,z)
        plt.show()"""  
    
if __name__=="__main__":
    plotyearwise()














