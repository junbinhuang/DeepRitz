import numpy as np 
import math, torch, generateData, time
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, StepLR
import torch.nn as nn
import matplotlib.pyplot as plt
import sys, os
import writeSolution

# Network structure
class RitzNet(torch.nn.Module):
    def __init__(self, params):
        super(RitzNet, self).__init__()
        self.params = params
        self.linearIn = nn.Linear(self.params["d"], self.params["width"])
        self.linear = nn.ModuleList()
        for _ in range(params["depth"]):
            self.linear.append(nn.Linear(self.params["width"], self.params["width"]))

        self.linearOut = nn.Linear(self.params["width"], self.params["dd"])

    def forward(self, x):
        x = torch.tanh(self.linearIn(x)) # Match dimension
        for layer in self.linear:
            x_temp = torch.tanh(layer(x))
            x = x_temp
        
        return self.linearOut(x)

def preTrain(model,device,params,preOptimizer,preScheduler,fun):
    model.train()
    file = open("lossData.txt","w")

    for step in range(params["preStep"]):
        # The volume integral
        data = torch.from_numpy(generateData.sampleFromDisk(params["radius"],params["bodyBatch"])).float().to(device)

        output = model(data)

        target = fun(params["radius"],data)

        loss = output-target
        loss = torch.mean(loss*loss)*math.pi*params["radius"]**2

        if step%params["writeStep"] == params["writeStep"]-1:
            with torch.no_grad():
                ref = exact(params["radius"],data)
                error = errorFun(output,ref,params)
                # print("Loss at Step %s is %s."%(step+1,loss.item()))
                print("Error at Step %s is %s."%(step+1,error))
            file.write(str(step+1)+" "+str(error)+"\n")

        model.zero_grad()
        loss.backward()

        # Update the weights.
        preOptimizer.step()
        # preScheduler.step()

def train(model,device,params,optimizer,scheduler):
    model.train()

    data1 = torch.from_numpy(generateData.sampleFromDisk(params["radius"],params["bodyBatch"])).float().to(device)
    data2 = torch.from_numpy(generateData.sampleFromSurface(params["radius"],params["bdryBatch"])).float().to(device)
    
    x_shift = torch.from_numpy(np.array([params["diff"],0.0])).float().to(device)
    y_shift = torch.from_numpy(np.array([0.0,params["diff"]])).float().to(device)
    data1_x_shift = data1+x_shift
    data1_y_shift = data1+y_shift

    for step in range(params["trainStep"]-params["preStep"]):
        output1 = model(data1)
        output1_x_shift = model(data1_x_shift)
        output1_y_shift = model(data1_y_shift)

        dfdx = (output1_x_shift-output1)/params["diff"] # Use difference to approximate derivatives.
        dfdy = (output1_y_shift-output1)/params["diff"] 

        model.zero_grad()

        # Loss function 1
        fTerm = ffun(data1).to(device)
        loss1 = torch.mean(0.5*(dfdx*dfdx+dfdy*dfdy)-fTerm*output1) * math.pi*params["radius"]**2

        # Loss function 2
        output2 = model(data2)
        target2 = exact(params["radius"],data2)
        loss2 = torch.mean((output2-target2)*(output2-target2) * params["penalty"] * 2*math.pi*params["radius"])
        loss = loss1+loss2                  

        if step%params["writeStep"] == params["writeStep"]-1:
            with torch.no_grad():
                target = exact(params["radius"],data1)
                error = errorFun(output1,target,params)
                # print("Loss at Step %s is %s."%(step+params["preStep"]+1,loss.item()))
                print("Error at Step %s is %s."%(step+params["preStep"]+1,error))
            file = open("lossData.txt","a")
            file.write(str(step+params["preStep"]+1)+" "+str(error)+"\n")

        if step%params["sampleStep"] == params["sampleStep"]-1:
            data1 = torch.from_numpy(generateData.sampleFromDisk(params["radius"],params["bodyBatch"])).float().to(device)
            data2 = torch.from_numpy(generateData.sampleFromSurface(params["radius"],params["bdryBatch"])).float().to(device)

            data1_x_shift = data1+x_shift
            data1_y_shift = data1+y_shift

        if 10*(step+1)%params["trainStep"] == 0:
            print("%s%% finished..."%(100*(step+1)//params["trainStep"]))

        loss.backward()

        optimizer.step()
        scheduler.step()      

def errorFun(output,target,params):
    error = output-target
    error = math.sqrt(torch.mean(error*error)*math.pi*params["radius"]**2)
    # Calculate the L2 norm error.
    ref = math.sqrt(torch.mean(target*target)*math.pi*params["radius"]**2)
    return error/ref   

def test(model,device,params):
    numQuad = params["numQuad"]

    data = torch.from_numpy(generateData.sampleFromDisk(1,numQuad)).float().to(device)
    output = model(data)
    target = exact(params["radius"],data).to(device)

    error = output-target
    error = math.sqrt(torch.mean(error*error)*math.pi*params["radius"]**2)
    # Calculate the L2 norm error.
    ref = math.sqrt(torch.mean(target*target)*math.pi*params["radius"]**2)
    return error/ref

def ffun(data):
    # f = 4
    return 4.0*torch.ones([data.shape[0],1],dtype=torch.float)

def exact(r,data):
    # f = 4 ==> u = r^2-x^2-y^2
    output = r**2-torch.sum(data*data,dim=1)

    return output.unsqueeze(1)

def rough(r,data):
    # A rough guess
    output = r**2-r*torch.sum(data*data,dim=1)**0.5
    return output.unsqueeze(1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def main():
    # Parameters
    # torch.manual_seed(21)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    params = dict()
    params["radius"] = 1
    params["d"] = 2 # 2D
    params["dd"] = 1 # Scalar field
    params["bodyBatch"] = 1024 # Batch size
    params["bdryBatch"] = 1024 # Batch size for the boundary integral
    params["lr"] = 0.01 # Learning rate
    params["preLr"] = 0.01 # Learning rate (Pre-training)
    params["width"] = 8 # Width of layers
    params["depth"] = 2 # Depth of the network: depth+2
    params["numQuad"] = 40000 # Number of quadrature points for testing
    params["trainStep"] = 50000
    params["penalty"] = 500
    params["preStep"] = 0
    params["diff"] = 0.001
    params["writeStep"] = 50
    params["sampleStep"] = 10
    params["step_size"] = 5000
    params["gamma"] = 0.3
    params["decay"] = 0.00001

    startTime = time.time()
    model = RitzNet(params).to(device)
    print("Generating network costs %s seconds."%(time.time()-startTime))

    preOptimizer = torch.optim.Adam(model.parameters(),lr=params["preLr"])
    optimizer = torch.optim.Adam(model.parameters(),lr=params["lr"],weight_decay=params["decay"])
    scheduler = StepLR(optimizer,step_size=params["step_size"],gamma=params["gamma"])

    startTime = time.time()
    preTrain(model,device,params,preOptimizer,None,rough)
    train(model,device,params,optimizer,scheduler)
    print("Training costs %s seconds."%(time.time()-startTime))

    model.eval()
    testError = test(model,device,params)
    print("The test error (of the last model) is %s."%testError)
    print("The number of parameters is %s,"%count_parameters(model))

    torch.save(model.state_dict(),"last_model.pt")

    pltResult(model,device,100,params)

def pltResult(model,device,nSample,params):
    rList = np.linspace(0,params["radius"],nSample)
    thetaList = np.linspace(0,math.pi*2,nSample)

    xx = np.zeros([nSample,nSample])
    yy = np.zeros([nSample,nSample])
    zz = np.zeros([nSample,nSample])
    for i in range(nSample):
        for j in range(nSample):
            xx[i,j] = rList[i]*math.cos(thetaList[j])
            yy[i,j] = rList[i]*math.sin(thetaList[j])
            coord = np.array([xx[i,j],yy[i,j]])
            zz[i,j] = model(torch.from_numpy(coord).float().to(device)).item()
            # zz[i,j] = params["radius"]**2-xx[i,j]**2-yy[i,j]**2 # Plot the exact solution.
    
    file = open("nSample.txt","w")
    file.write(str(nSample))

    file = open("Data.txt","w")
    writeSolution.write(xx,yy,zz,nSample,file)

    edgeList = [[params["radius"]*math.cos(i),params["radius"]*math.sin(i)] for i in thetaList]
    writeSolution.writeBoundary(edgeList)

if __name__=="__main__":
    main()