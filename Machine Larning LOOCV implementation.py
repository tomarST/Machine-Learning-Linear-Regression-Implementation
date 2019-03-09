import numpy as np
import matplotlib.pyplot as plt
def polynomial(f,n):
    s = np.size(f[:,0]) #number of sample observations
    x = f[:,0]          #attribute values: a row vector
    t = f[:,1]          #target values: a row vector
    u = np.ones((1,s))  #a row vector of all 1s
    for i in range(1,n+1):
        u = np.vstack((u,np.power(x,i)))
    return u,t
def LOOCV(f,n): 
    X,t2=polynomial(f,n)
    X=X.T
    sums=[]
    small_LOCCV_M=0
    for i in range(1,n+1):
        x1=np.copy(X)
        x1=x1[:,:i+1]
        t1=np.copy(t2)
        sum1=0
        for v in range(0,np.size(t2)):
            x_i=np.vstack((x1[:v,:],x1[v+1:,:]))
            t_i=np.hstack((t1[:v],t1[v+1:]))
            v_i=np.linalg.inv(x_i.T@x_i)
            w_i=v_i@x_i.T@t_i
            sum1=sum1+np.power((x1[v]@w_i.T)-t1[v],2)
        sums.append(sum1/np.size(t2))
    min1=min(sums)
    order=sums.index(min1)
    return sums,min1,order+1
def plotit(f,n): #Plot a graph of Order vs Mean Loss
    x3=np.arange(1,n+1)
    y3,i,j=LOOCV(f,n)
    plt.plot(x3,y3)
    plt.xlabel("Polynomial Order")
    plt.ylabel("Mean Loss")
    plt.show()


#Main program
     
f=np.loadtxt(r"C:\Users\Soham\Downloads\synthdata.txt")
n=8
polynomial(f,n)
sums,min_MeanLoss,order=LOOCV(f,n)
plotit(f,n)
for j in sums:
        print("Order:",sums.index(j)+1,"and Mean Loss:",j)
print("Order of polynomial with minimum mean loss:",order,"and Minimum of mean loss:",min_MeanLoss)
u,t=polynomial(f,order)
print("Coefficients of polynomial with minimum LOOCV Loss:",np.linalg.inv(u@u.T)@u@t)
