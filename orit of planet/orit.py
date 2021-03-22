import numpy as np
import matplotlib.pyplot as plt

def cal():
    """
        the ode is as list:
            x'' = -G*M*x/r^3
            y'' = -G*M*y/r^3
            x^2+y^2 = r^2

        and :
            x(t+dt) = x(t) + dt*v(t+dt/2)  mid fdm
            v(t+dt/2) = v(t-dt/2) + dt*a(t)
            a(t) = -G*M*x/r^3

            v(dt/2) = v(0) + dt/2*a(0)

    """
    
    n = 400000
    dt = 0.01
    t = np.linspace(0,dt*n,n+1)

    x = np.linspace(0,1,n+1)
    y = np.linspace(0,1,n+1)
    r = np.linspace(0,1,n+1)

    v_x = np.linspace(0,1,n+1)
    v_y = np.linspace(0,1,n+1)

    a_x = np.linspace(0,1,n+1)
    a_y = np.linspace(0,1,n+1)

    # initial condition
    
    x[0] = 0.5791
    y[0] = 0
    r[0] = 0.5791
   
    v_x[0] = 0
    v_y[0] = 1.73
   
    a_x[0] = ax(x[0],y[0])
    a_y[0] = 0

    v_x[1] = v_x[0] + dt/2*a_x[0]
    v_y[1] = v_y[0] + dt/2*a_y[0]
   
    for i in range(n):
        if i!=0:  #skip initial conditon
            v_x[i+1] = v_x[i] + dt*a_x[i]
            v_y[i+1] = v_y[i] + dt*a_y[i]
        x[i+1] = x[i] + dt*v_x[i+1]
        y[i+1] = y[i] + dt*v_y[i+1]
       

        a_x[i+1] = ax(x[i+1],y[i+1])
        a_y[i+1] = ay(x[i+1],y[i+1])

    data = {
                "n" : n,
                "t" : t,
                "dt" : dt,
                "x" : x,
                "y" : y,
                "r" : r,
                "v_x" : v_x,
                "v_y" : v_y,
                "a_x" : a_x,
                "a_y" : a_y }
    
    return data
    
def ax(x,y):
    """
       ax(t) = -G*M*x/r^3
    """ 
    G=1;M=1
    k = -1*G*M
    r = pow((pow(x,2)+pow(y,2)),0.5)
    s = pow(r,3)
    s = x/s
    s = k*s
    return s 

def ay(x,y):
    """
       ay(t) = -G*M*y/r^3
    """ 
    G=1;M=1
    k = -1*G*M
    r = pow((pow(x,2)+pow(y,2)),0.5)
    s = pow(r,3)
    s = y/s
    s = k*s
    return s 

def plot(data):
    x = data["x"]
    y = data["y"]
    plt.plot(x,y)
    plt.xlabel('x label')
    plt.ylabel('y label')
    plt.title('orit of planet')
    plt.show()

def fit(data):
    x = data["x"]
    y = data["y"]
    m = len(x)
    x2 = np.power(x,2)
    y2 = np.power(y,2)
    xy = np.dot(x,y)
    X = np.zeros([m,5])
    Y = np.ones([m,1])*-1
    X[:,0] = x2
    X[:,1] = xy
    X[:,2] = y2
    X[:,3] = x
    X[:,4] = y

    A = np.dot(X.T,X)
    A = np.linalg.inv(A)
    A = np.dot(A,X.T)
    A = np.dot(A,Y)

    return A
data = cal()
A = fit(data)
print(A)
plot(data)
