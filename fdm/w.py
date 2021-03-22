import numpy as np
import matplotlib.pyplot as plt
from math import *
def initialize():
    n = int(input('pls input the num of pointers : '))    
    L = 6.096
    h = L / n
    EI = 2.6336e9
    k = 5.0e7
    rank = 4
    q = 14880 

    x = np.linspace(0,L,n+1)
    D = np.zeros([n+1-rank,n+1])
    b = np.zeros([n+1,1])

    for i in range(n+1-rank):
        D[i,i] = 1
        D[i,i+1] = -4
        D[i,i+2] = 6 + k*pow(h,4)/EI
        D[i,i+3] = -4
        D[i,i+4] = 1

    A = np.pad(D,(2,2))
    A = np.delete(A,[0,1],axis=1)
    A = np.delete(A,-1,axis=1)
    A = np.delete(A,-1,axis=1)
    
    for i in range(n):
        b[i,0] = q*pow(h,4)/EI
    assert(b.shape == (n+1,1))
    
    # boundray condition

    A[0,0] = 1 ; b[0,0] = 0
    A[1,0] = 1 ;A[1,1] = -2; A[1,2] = 1 ; b[1,0] = 0
    A[-1,-1] = 1 ; b[-1,0] = 0
    A[-2,-1] = 1;A[-2,-2] = -2;A[-2,-3] = 1; b[-2,0] = 0

    print(A)
    return A,b,x



if __name__ == '__main__':

    A,b,x = initialize()
    w = np.linalg.solve(A,b)
    y = compare(x)
    # v_force => f
    k = 5.0e7
    f = k*w
    # M 


    plt.plot(x,w)
    plt.show()