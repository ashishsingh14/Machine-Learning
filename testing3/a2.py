import sys, os
import csv
def converttocsv1():
    c = open("mumps.csv", "wb")
    d = csv.writer(c, delimiter=',')
    old = ''
    with open("MUMPS_Cases_1968-2003_20160414022715.csv", 'r') as f:
        for line in f:
            a =  line.split()
            
            if a[2] == '58' and a[0]!=old :
                
                new_list = []
                new_list.append(a[0])
                new_list.append(a[3])
                d.writerow(new_list)
                old = a[0]
                
def converttocsv2():
    c = open("mumps.csv", "wb")
    d = csv.writer(c, delimiter=',')
    with open("MUMPS_Cases_1968-2003_20160414022715.csv", 'r') as f:
        f.readline()
        f.readline()
        f.readline()
        flag = 0
        while True:         
            new = []
            sum = 0        
            for i in range(52):
                e = f.readline()
                if e =='':
                    flag = 1
                    break
                else:
                    a = e.split(',')
                print a
                
                if i == 0:
                    new.append(a[0])
                    afterstrip = a[2].strip("\n")
                    if afterstrip.isdigit():
                        sum = sum + int(a[2])                    
                else:
                    afterstrip = a[2].strip("\n")
                    if afterstrip.isdigit():
                        sum = sum + int(a[2]) 
            if flag == 1:
                break
            print sum        
            new.append(sum)
            d.writerow(new)



if __name__=="__main__":            
    a = '122'
    b = '133'
    c = float(a + '.' +  b)
    print type(c)















