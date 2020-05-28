import numpy as np
import math
import matplotlib.pyplot as plt

# Sample points in a disk
def sampleFromDisk(r,n):
    """
    r -- radius;
    n -- number of samples.
    """
    array = np.random.rand(2*n,2)*2*r-r
    
    array = np.multiply(array.T,(np.linalg.norm(array,2,axis=1)<r)).T
    array = array[~np.all(array==0, axis=1)]
    
    if np.shape(array)[0]>=n:
        return array[0:n]
    else:
        return sampleFromDisk(r,n)

def sampleFromDomain(n):
    # For simplicity, consider a square with a hole.
    # Square: [-1,1]*[-1,1]
    # Hole: c = (0.3,0.0), r = 0.3
    array = np.zeros([n,2])
    c = np.array([0.3,0.0])
    r = 0.3

    for i in range(n):
        array[i] = randomPoint(c,r)

    return array

def randomPoint(c,r):
    point = np.random.rand(2)*2-1
    if np.linalg.norm(point-c)<r:
        return randomPoint(c,r)
    else:
        return point

def sampleFromBoundary(n):
    # For simplicity, consider a square with a hole.
    # Square: [-1,1]*[-1,1]
    # Hole: c = (0.3,0.0), r = 0.3
    c = np.array([0.3,0.0])
    r = 0.3
    length = 4*2+2*math.pi*r
    interval1 = np.array([0.0,2.0/length])
    interval2 = np.array([2.0/length,4.0/length])
    interval3 = np.array([4.0/length,6.0/length])
    interval4 = np.array([6.0/length,8.0/length])
    interval5 = np.array([8.0/length,1.0])

    array = np.zeros([n,2])

    for i in range(n):
        rand0 = np.random.rand()
        rand1 = np.random.rand()

        point1 = np.array([rand1*2.0-1.0,-1.0])
        point2 = np.array([rand1*2.0-1.0,+1.0])
        point3 = np.array([-1.0,rand1*2.0-1.0])
        point4 = np.array([+1.0,rand1*2.0-1.0])
        point5 = np.array([c[0]+r*math.cos(2*math.pi*rand1),c[1]+r*math.sin(2*math.pi*rand1)])

        array[i] = myFun(rand0,interval1)*point1 + myFun(rand0,interval2)*point2 + \
            myFun(rand0,interval3)*point3 + myFun(rand0,interval4)*point4 + \
                myFun(rand0,interval5)*point5
 
    return array

def myFun(x,interval):
    if interval[0] <= x <= interval[1]:
        return 1.0
    else: return 0.0

def sampleFromSurface(r,n):
    """
    r -- radius;
    n -- number of samples.
    """
    array = np.random.normal(size=(n,2))
    norm = np.linalg.norm(array,2,axis=1)
    # print(np.min(norm))
    if np.min(norm) == 0:
        return sampleFromSurface(r,n)
    else:
        array = np.multiply(array.T,1/norm).T
        return array*r

# Sample from 10d-ball
def sampleFromDisk10(r,n):
    """
    r -- radius;
    n -- number of samples.
    """
    array = np.random.normal(size=(n,10))
    norm = np.linalg.norm(array,2,axis=1)
    # print(np.min(norm))
    if np.min(norm) == 0:
        return sampleFromDisk10(r,n)
    else:
        array = np.multiply(array.T,1/norm).T
        radius = np.random.rand(n,1)**(1/10)
        array = np.multiply(array,radius)

        return r*array

def sampleFromSurface10(r,n):
    """
    r -- radius;
    n -- number of samples.
    """
    array = np.random.normal(size=(n,10))
    norm = np.linalg.norm(array,2,axis=1)
    # print(np.min(norm))
    if np.min(norm) == 0:
        return sampleFromSurface10(r,n)
    else:
        array = np.multiply(array.T,1/norm).T
        return array*r

if __name__ == "__main__":
    # array = sampleFromDomain(10000).T
    # array = sampleFromBoundary(500).T
    # plt.plot(array[0],array[1],'o',ls="None")
    # plt.axis("equal")
    # plt.show()
    pass