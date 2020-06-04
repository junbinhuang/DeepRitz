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
                ref = exact(data)
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
    ratio = (4*2.0+2*math.pi*0.3)/(2.0*2.0-math.pi*0.3**2)
    model.train()

    data1 = torch.from_numpy(generateData.sampleFromDomain(params["bodyBatch"])).float().to(device)
    data1.requires_grad = True
    data2 = torch.from_numpy(generateData.sampleFromBoundary(params["bdryBatch"])).float().to(device)

    for step in range(params["trainStep"]-params["preStep"]):
        output1 = model(data1)

        model.zero_grad()

        dfdx = torch.autograd.grad(output1,data1,grad_outputs=torch.ones_like(output1),retain_graph=True,create_graph=True,only_inputs=True)[0]
        dfdxx = torch.autograd.grad(dfdx[:,0].unsqueeze(1),data1,grad_outputs=torch.ones_like(output1),retain_graph=True,create_graph=True,only_inputs=True)[0][:,0].unsqueeze(1)
        dfdyy = torch.autograd.grad(dfdx[:,1].unsqueeze(1),data1,grad_outputs=torch.ones_like(output1),retain_graph=True,create_graph=True,only_inputs=True)[0][:,1].unsqueeze(1)
        # Loss function 1
        fTerm = ffun(data1).to(device)
        loss1 = torch.mean((dfdxx+dfdyy+fTerm)*(dfdxx+dfdyy+fTerm))

        # Loss function 2
        output2 = model(data2)
        target2 = exact(data2)
        loss2 = torch.mean((output2-target2)*(output2-target2) * params["penalty"] * ratio)
        loss = loss1+loss2            

        if step%params["writeStep"] == params["writeStep"]-1:
            with torch.no_grad():
                target = exact(data1)
                error = errorFun(output1,target,params)
                # print("Loss at Step %s is %s."%(step+params["preStep"]+1,loss.item()))
                print("Error at Step %s is %s."%(step+params["preStep"]+1,error))
            file = open("lossData.txt","a")
            file.write(str(step+params["preStep"]+1)+" "+str(error)+"\n")

        if step%params["sampleStep"] == params["sampleStep"]-1:
            data1 = torch.from_numpy(generateData.sampleFromDomain(params["bodyBatch"])).float().to(device)
            data1.requires_grad = True
            data2 = torch.from_numpy(generateData.sampleFromBoundary(params["bdryBatch"])).float().to(device)

        if 10*(step+1)%params["trainStep"] == 0:
            print("%s%% finished..."%(100*(step+1)//params["trainStep"]))

        loss.backward()

        optimizer.step()
        scheduler.step()

def errorFun(output,target,params):
    error = output-target
    error = math.sqrt(torch.mean(error*error))
    # Calculate the L2 norm error.
    ref = math.sqrt(torch.mean(target*target))
    return error/ref   

def test(model,device,params):
    numQuad = params["numQuad"]

    data = torch.from_numpy(generateData.sampleFromDomain(numQuad)).float().to(device)
    output = model(data)
    target = exact(data).to(device)

    error = output-target
    error = math.sqrt(torch.mean(error*error))
    # Calculate the L2 norm error.
    ref = math.sqrt(torch.mean(target*target))
    return error/ref

def ffun(data):
    # f = 0.0
    return 0.0*torch.ones([data.shape[0],1],dtype=torch.float)

def exact(data):
    # f = 0 ==> u = xy
    output = data[:,0]*data[:,1]

    return output.unsqueeze(1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

# def rough(r,data):
#     output = r**2-r*torch.sum(data*data,dim=1)**0.5
#     return output.unsqueeze(1)

def main():
    # Parameters
    # torch.manual_seed(21)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    params = dict()
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
    params["writeStep"] = 50
    params["sampleStep"] = 10
    params["step_size"] = 5000
    params["gamma"] = 0.5
    params["decay"] = 0.00001

    startTime = time.time()
    model = RitzNet(params).to(device)
    print("Generating network costs %s seconds."%(time.time()-startTime))

    preOptimizer = torch.optim.Adam(model.parameters(),lr=params["preLr"])
    optimizer = torch.optim.Adam(model.parameters(),lr=params["lr"],weight_decay=params["decay"])
    scheduler = StepLR(optimizer,step_size=params["step_size"],gamma=params["gamma"])

    startTime = time.time()
    preTrain(model,device,params,preOptimizer,None,exact)
    train(model,device,params,optimizer,scheduler)
    print("Training costs %s seconds."%(time.time()-startTime))

    model.eval()
    testError = test(model,device,params)
    print("The test error (of the last model) is %s."%testError)
    print("The number of parameters is %s,"%count_parameters(model))

    torch.save(model.state_dict(),"last_model.pt")

    pltResult(model,device,500,params)

def pltResult(model,device,nSample,params):
    xList = np.linspace(-1,1,nSample)
    yList = np.linspace(-1,1,nSample)
    thetaList = np.linspace(0,2*math.pi,50)

    xx = np.zeros([nSample,nSample])
    yy = np.zeros([nSample,nSample])
    zz = np.zeros([nSample,nSample])
    for i in range(nSample):
        for j in range(nSample):
            xx[i,j] = xList[i]
            yy[i,j] = yList[j]
            coord = np.array([xx[i,j],yy[i,j]])
            zz[i,j] = model(torch.from_numpy(coord).float().to(device)).item()
            # zz[i,j] = xx[i,j]*yy[i,j] # Plot the exact solution.
            if np.linalg.norm(coord-np.array([0.3,0.0]))<0.3:
                zz[i,j] = "NaN"
    
    file = open("nSample.txt","w")
    file.write(str(nSample))

    file = open("Data.txt","w")
    writeSolution.write(xx,yy,zz,nSample,file)

    edgeList2 = [[0.3*math.cos(i)+0.3,0.3*math.sin(i)] for i in thetaList]
    edgeList1 = [[-1.0,-1.0],[1.0,-1.0],[1.0,1.0],[-1.0,1.0],[-1.0,-1.0]]
    writeSolution.writeBoundary(edgeList1,edgeList2)

if __name__=="__main__":
    main()