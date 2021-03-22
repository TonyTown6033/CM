import numpy as np
import matplotlib.pyplot as plt
import yagmail

def sett():
    """
        '水星','金星','地球','火星','木星','土星','天王星','海王星'
    """
    pi = 3.14159265359
    GM = 6.67430e-11*1.989e30
    l = np.zeros([8,1])
    T = np.zeros([8,1])
    l[0] = 5791e7
    T[0] = 87.70*24*60*60
    l[1] = 10820e7
    T[1] = 224.701*24*60*60
    l[2] = 14960e7
    T[2] = 365.2422*24*60*60
    l[3] = 22794e7
    T[3] = 686.98*24*60*60
    l[4] = 77833e7
    T[4] = 4430*24*60*60
    l[5] = 142940e7
    T[5] = 29.46*365.2422*24*60*60
    l[6] = 287099e7
    T[6] = 84*365.2422*24*60*60
    l[7] = 450400e7
    T[7] = 165*365.2422*24*60*60
    w = np.divide(2*pi,T)
    v = np.multiply(w,l)
    return l,v

def cal(x0,v0):
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
    dt = 12000
    t = np.linspace(0,dt*n,n+1)

    x = np.linspace(0,1,n+1)
    y = np.linspace(0,1,n+1)
    r = np.linspace(0,1,n+1)

    v_x = np.linspace(0,1,n+1)
    v_y = np.linspace(0,1,n+1)

    a_x = np.linspace(0,1,n+1)
    a_y = np.linspace(0,1,n+1)

    # initial condition
    
    x[0] = x0
    y[0] = 0
    r[0] = x0
   
    v_x[0] = 0
    v_y[0] = v0
   
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
    G=6.67430e-11;M=1.989e30
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
    G=6.67430e-11;M=1.989e30
    k = -1*G*M
    r = pow((pow(x,2)+pow(y,2)),0.5)
    s = pow(r,3)
    s = y/s
    s = k*s
    return s 

def plot(data,i):
    x = data["x"]
    y = data["y"]
    planets = ['']
    plt.plot(x,y,label = str(i+1)+'th')
    plt.xlabel('x label')
    plt.ylabel('y label')
    plt.title('orit of planet')
    plt.legend()
    

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
def send():
    try:
        qqmail=['1026828424@qq.com','3024719017@qq.com','3488239190@qq.com','1471075816@qq.com','1030888182@qq.com','1005068490@qq.com','1310435374@qq.com','2354656987@qq.com','1329096811@qq.com','812654659@qq.com','460454587@qq.com','1357257440@qq.com','2869362377@qq.com','1310508360@qq.com','1184264060@qq.com','1041223219@qq.com','714007495@qq.com','1360278197@qq.com']
        title=['太阳系八大行星的运行轨迹']
        contents=['通过有限差分法算出。详细的程序见附件，输出的轨迹也见附件。\n来自TonyTown']
        yag=yagmail.SMTP(user='603368546@qq.com',password='uwuorxhvjealbefb',host='smtp.qq.com')
        yag.send(to=qqmail,subject=title,contents=contents,attachments=['D:/Code/CM/orit of planet/8planets.jpg','D:/Code/CM/orit of planet/4planets.jpg','D:/Code/CM/orit of planet/plants.py'])
        print('Email send success')
    except:
        print('Email send fail')

x0,v0 = sett()
for i in range(8):

    data = cal(x0[i],v0[i])
    A = fit(data)
    plot(data,i)
    
    
plt.savefig('D:/Code/CM/orit of planet/8planets.jpg')
plt.show()
for i in range(4):
    
    data = cal(x0[i],v0[i])
    A = fit(data)
    plot(data,i)
    
    
plt.savefig('D:/Code/CM/orit of planet/4planets.jpg')
plt.show()
send()