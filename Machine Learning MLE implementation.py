import numpy as np
import matplotlib.pyplot as plt
def polynomial(f,n):
    s = np.size(f[:,0]) #number of sample observations
    x = f[:,0]          #attribute values: a row vector
    S=min(x)
    x=x-S
    t = f[:,1]          #target values: a row vector
    u = np.ones((1,s))  #a row vector of all 1s
    for i in range(1,n+1):
        u = np.vstack((u,np.power(x,i)))
    return u,t
def w(f,n):
    u,t=polynomial(f,n)
    v = np.linalg.inv(u@u.T) #v = (uu')^-1
    w = v@u@t           #w = vut
    return(w)
def var(f,n):
    u,t=polynomial(f,n)
    w0=w(f,n)
    var=(np.dot((t-np.dot(u.T,w0)).T,(t-np.dot(u.T,w0))))/(len(t))
    return var
    
def log_L(f,n):
    log_Like=[]
    variance=[]
    order=[]
    for v in range(1,n+1):
        sum1=0
        u,t=polynomial(f,v)
        a=(len(t))*np.log(2*np.pi)/2
        u=u.T
        w0=w(f,v)
        v0=np.sqrt(var(f,v))
        (f,v)
        b=(len(t))*np.log(v0)
        for i in range(len(t)):
            x0=u[i,:]
            g=x0@w0
            diff=(t[i]-g)**2
            sum1=sum1+diff
        c=(sum1)/(2*(v0)**2)
        l=-a-b-c
        
        order.append(v)
        log_Like.append(l)
        variance.append(v0**2)
    return order,log_Like,variance
def plotit(f,n):
    o,l,v=log_L(f,n)
    c=max(l)/max(v)
    for i in range(len(v)):
        v[i]=c*v[i]
    plt.plot(o,l,'-o',label="Loglikelihood")
    plt.plot(o,v,"-o",label="Variance")
    plt.legend()
    plt.xlabel("Order")
    plt.ylabel("Bluedot: loglikelihood and Line: sigma^2")
    plt.show()
    
f=np.loadtxt(r"C:\Users\Soham\Desktop\Olympics.txt")
O,L,V=log_L(f,5)
plotit(f,5)
for i in range(len(O)):
    print("Order:",O[i],", Log Likelihood:",L[i],", Variance",np.sqrt(V[i]))
