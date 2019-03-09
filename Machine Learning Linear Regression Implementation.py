

import numpy as np
import matplotlib.pyplot as plt

#This function fits a linear function to the data. The data
#is stored in matrix variable f where the 1st column is 
#attribute values and the 2nd column is target values. The
#function returns the co-efficients

def linear(f):
	s = np.size(f[:,0]) #number of sample observations
	x = f[:,0]          #attribute values: a row vector
	t = f[:,1]          #target values: a row vector
	u = np.ones((1,s))  #a row vector of all 1s
	u = np.vstack((u,x))#the transpose of the data matrix shown in the class
	v = np.linalg.inv(u@u.T) #v = (uu')^-1
	w = v@u@t           #w = vut
	return(w)

#This function plots the linear function with coefficient vector w
#against the attribue values. It also plots the data as the dots.
#The data is stored in matrix variable f where the 1st column is
#attribute values and the 2nd column is target values. 
def predict(w,x):
    t=w@x
    return t
def plotit(w, f):
        s = np.size(f[:,0])#These five lines are the same as in linear(f)
        x = f[:,0]
        t = f[:,1]
        u = np.ones((1,s))
        u = np.vstack((u,x))
        t1 = w@u
        plt.plot(x,t,'o', x,t1)
        plt.xlabel('years')
        plt.ylabel('winning times')
        plt.xlim(1920,2020)
        plt.show()

f=np.loadtxt(r"C:\Users\Soham\Desktop\Olympics.txt")
w=linear(f)
print("Coefficients",w.T)
plotit(w,f)
x=np.array([[1,1],[2012,2016]])
y=predict(w,x)
print("Predicted target values for 2012 and 2016",y)
