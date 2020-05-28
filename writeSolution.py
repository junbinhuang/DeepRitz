import sys, os
import numpy as np
import math

def writeRow(list,file):
    for i in list: file.write("%s "%i)
    file.write("\n")

def write(X,Y,Z,nSampling,file):
    for k1 in range(nSampling):
        writeRow(X[k1],file)
        writeRow(Y[k1],file)
        writeRow(Z[k1],file)

def writeBoundary(edgeList,edgeList2 = None):
    length=[]
    file=open("boundaryCoord.txt","w")

    for i in edgeList:
        writeRow(i,file)
    if edgeList2 != None:
        for i in edgeList2:
            writeRow(i,file)

    file=open("boundaryNumber.txt","w")
    if edgeList2 == None: length = [len(edgeList)]
    else: length = [len(edgeList),len(edgeList2)]

    for i in length:
        file.write("%s\n"%i)

if __name__=="__main__":
    pass