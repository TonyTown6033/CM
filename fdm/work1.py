import numpy as np
import matplotlib.pyplot as plt
import os 

n = 5
h = 1 / n 
x = np.arange(0,1+h,h)
A = np.zeros([n+1,n+1])

for i in range(n):
    A[i][i] = 1 
    if i+1<n+1:
        A[i][i+1] = -3
    if i+2<n+1:
        A[i][i+2] = 1

A = np.delete(A,[-1,-2],axis=0)
A = np.pad(A,((1,1),(0,0)),'constant', constant_values=(0,0))
A[0][0] = 1
A[-1][-1] = 1


def f(x):
    """
        f(x) = [(x + sinx)/(1+x)]^2
    """
    a = np.sin(x)+x
    b = 1 + x
    s = np.divide(a,b)
    s = np.power(s,2)

    return s 

s = f(x)
A = np.pad(A,((0,0),(0,1)),'constant', constant_values=(0,0))


for i in range(n+1):
    A[i][-1] = s[i]

A[-1][-1] = 1
with open ('data.txt','w') as f:
    for i in range(len(A)):
        for j in range(len(A)+1):
            str1 = str(A[i][j])
            f.write(str1)
            f.write(' ')
        f.write('\n')

os.system("pause")

outtemp = []
for line in open('out.txt'):
    outtemp.append(line)

y = np.zeros([5,1])
for i in range(5):
    y[i] = outtemp[i]


fig = plt.figure()
plt.plot(x,y)
plt.show()

