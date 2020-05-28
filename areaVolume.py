# The surface area of n-dimensional sphere.
import numpy as np 
import math

def areaVolume(r,n):
    area = np.zeros(n)
    volume = np.zeros(n)

    area[0] = 0
    area[1] = 2
    area[2] = 2*math.pi*r
    volume[0] = 1
    volume[1] = 2*r
    volume[2] = math.pi*r**2

    for i in range(3,n):
        area[i] = 2*area[i-2]*math.pi*r**2 / (i-2)
        volume[i] = 2*math.pi*volume[i-2]*r**2 / i

    return (area[-1]/volume[-1])

if __name__=="__main__":
    n = 10 # Number of dimensions
    r = 1.0 # Radius
    areaVolume(r,n)
