import numpy as np
import matplotlib.pyplot as plt
import yagmail

def inialize():
    # y'' + p(x)*y' + q(x)*y = r(x) 
    n = int(input('pls input the number of point: '))
    h = 1 / n
    rank = 2
    x = np.linspace(0,1,n+1)
    A = np.zeros([n+1,n+1])

    # diag(a_i,b_i,c_i)
    a = np.zeros([1,n+1-rank])
    b = np.zeros([1,n+1-rank])
    c = np.zeros([1,n+1-rank])
   
    a = 2*np.ones([1,n+1-rank]) - np.multiply(h,p(x))
    b = -4*np.ones([1,n+1-rank]) + np.multiply(2*h*h,q(x))
    c = 2*np.ones([1,n+1-rank]) + np.multiply(h,p(x))
    
    for i in range(n+1-rank):
        A[i][i] = a[0,i]
        if i+1<n+1:
            A[i][i+1] = b[0,i]
        if i+2<n+1:
            A[i][i+2] = c[0,i]

    A = np.delete(A,[-1,-2],axis=0)
    # padding Matrax
    A = np.pad(A,((1,1),(0,0)),'constant', constant_values=(0,0))
    A = np.pad(A,((0,0),(0,1)),'constant', constant_values=(0,0))

    # Constant Colunm
    s = r(x)
    s = np.multiply(s,h*h*2)
    A[:,-1] = s


    # Boundry Condition
    A[0,0] = 1 ;A[0,-1] = 0
    A[-1,-2] = 1 ;A[-1,-1] = 1


    return A,n,x

def r(x):
    """
        r(x) = [(x + sinx)/(1+x)]^2
    """
    a = np.sin(x)+x
    b = 1 + x
    s = np.divide(a,b)
    s = np.power(s,2)
    return s

def p(x):
    """
        y' parameter
    """
    s = 0

    return s 

def q(x):
    """
        y parameter
    """

    s = -1

    return s 

def send():
    try:
        yag=yagmail.SMTP(user='1026828424@qq.com',password='loxflcmgqpoqbcjg',host='smtp.qq.com')
        yag.send(to='603368546@qq.com',subject='test',contents='Hello',attachments=[r'D:/Code/CM/fdm/out.jpg'])
        print('Email send success')
    except:
        print('Email send fail')


if __name__ == "__main__":
    for i in range(3):
        A,n,x= inialize()

        A1 = A[:,0:-1]
        b = A[:,-1].T

        y = np.linalg.solve(A1,b)



    #    print('分割的点为： ')
    #    print(x)
    #    print('对应点上的值为')
    #    print(y)
        plt.plot(x,y,label = str(n)+' func')
        plt.xlabel('x axis')
        plt.ylabel('y axis')
        plt.title('FDM plot')
        plt.legend()
    
    plt.savefig('D:/Code/CM/fdm/out.jpg')
    plt.show()
    send()
