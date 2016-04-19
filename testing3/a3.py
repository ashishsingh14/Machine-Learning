import os, sys
import matplotlib.pyplot as plt

def plottrendanalysis():
    with open("MUMPS_Cases_1968-2003_20160414022715.csv", 'r') as f:
        f.readline()
        f.readline()
        f.readline()
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
    #print len(year_week)
    #print len(cases)
    plt.plot(year_week, cases)
    plt.xlim(1965, 2003)
    plt.title("Trend Analysis over the time")
    plt.ylabel("Year with corresponding week")
    plt.xlabel("Cases in Particular week")
    plt.grid(True)
    plt.show()


if __name__=="__main__":
    plottrendanalysis()          