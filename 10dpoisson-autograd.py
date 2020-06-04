import numpy as np 
import math, torch, generateData, time
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, StepLR, MultiplicativeLR
import torch.nn as nn
import matplotlib.pyplot as plt
import sys, os
import writeSolution
from areaVolume import areaVolume

# Network structure
class RitzNet(torch.nn.Module):
    def __init__(self, params):
        super(RitzNet, self).__init__()
        self.params = params
        # self.linearIn = nn.Linear(self.params["d"], self.params["width"])
        self.linear = nn.ModuleList()
        for _ in range(params["depth"]):
            self.linear.append(nn.Linear(self.params["width"], self.params["width"]))

        self.linearOut = nn.Linear(self.params["width"], self.params["dd"])

    def forward(self, x):
        # x = F.softplus(self.linearIn(x)) # Match dimension
        for layer in self.linear:
            x_temp = F.softplus(layer(x))
            x = x_temp+x
        
        return self.linearOut(x)

def initWeights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)

def preTrain(model,device,params,preOptimizer,preScheduler,fun):
    model.train()
    file = open("lossData.txt","w")

    for step in range(params["preStep"]):
        # The volume integral
        data = torch.from_numpy(generateData.sampleFromDisk10(params["radius"],params["bodyBatch"])).float().to(device)

        output = model(data)

        target = fun(params["radius"],data)

        loss = output-target
        loss = torch.mean(loss*loss)

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

    data1 = torch.from_numpy(generateData.sampleFromDisk10(params["radius"],params["bodyBatch"])).float().to(device)
    data1.requires_grad = True
    data2 = torch.from_numpy(generateData.sampleFromSurface10(params["radius"],params["bdryBatch"])).float().to(device)

    for step in range(params["trainStep"]-params["preStep"]):
        output1 = model(data1)

        model.zero_grad()

        dfdx = torch.autograd.grad(output1,data1,grad_outputs=torch.ones_like(output1),retain_graph=True,create_graph=True,only_inputs=True)[0]
        # Loss function 1
        fTerm = ffun(data1).to(device)
        loss1 = torch.mean(0.5*torch.sum(dfdx*dfdx,1).unsqueeze(1)-fTerm*output1)

        # Loss function 2
        output2 = model(data2)
        target2 = exact(params["radius"],data2)
        loss2 = torch.mean((output2-target2)*(output2-target2) * params["penalty"] *params["area"])
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
            data1 = torch.from_numpy(generateData.sampleFromDisk10(params["radius"],params["bodyBatch"])).float().to(device)
            data1.requires_grad = True
            data2 = torch.from_numpy(generateData.sampleFromSurface10(params["radius"],params["bdryBatch"])).float().to(device)

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

    data = torch.from_numpy(generateData.sampleFromDisk10(1,numQuad)).float().to(device)
    output = model(data)
    target = exact(params["radius"],data).to(device)

    error = output-target
    error = math.sqrt(torch.mean(error*error))
    # Calculate the L2 norm error.
    ref = math.sqrt(torch.mean(target*target))
    return error/ref

def ffun(data):
    # f = 0
    return 0.0*torch.ones([data.shape[0],1],dtype=torch.float)
    # f = 20
    # return 20.0*torch.ones([data.shape[0],1],dtype=torch.float)

def exact(r,data):
    # f = 20 ==> u = r^2-x^2-y^2-...
    # output = r**2-torch.sum(data*data,dim=1)
    # f = 0 ==> u = x1x2+x3x4+x5x6+...
    output = data[:,0]*data[:,1] + data[:,2]*data[:,3] + data[:,4]*data[:,5] + \
        data[:,6]*data[:,7] + data[:,8]*data[:,9]
    return output.unsqueeze(1)

def rough(r,data):
    # output = r**2-r*torch.sum(data*data,dim=1)**0.5
    output = torch.zeros(data.shape[0],dtype=torch.float)
    return output.unsqueeze(1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters()) # if p.requires_grad

def main():
    # Parameters
    torch.manual_seed(21)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    params = dict()
    params["radius"] = 1
    params["d"] = 10 # 10D
    params["dd"] = 1 # Scalar field
    params["bodyBatch"] = 1024 # Batch size
    params["bdryBatch"] = 2048 # Batch size for the boundary integral
    params["lr"] = 0.016 # Learning rate
    params["preLr"] = params["lr"] # Learning rate (Pre-training)
    params["width"] = 10 # Width of layers
    params["depth"] = 4 # Depth of the network: depth+2
    params["numQuad"] = 40000 # Number of quadrature points for testing
    params["trainStep"] = 50000
    params["penalty"] = 500
    params["preStep"] = 0
    params["writeStep"] = 50
    params["sampleStep"] = 10
    params["area"] = areaVolume(params["radius"],params["d"])
    params["step_size"] = 5000
    params["milestone"] = [5000,10000,20000,35000,48000]
    params["gamma"] = 0.5
    params["decay"] = 0.0001

    startTime = time.time()
    model = RitzNet(params).to(device)
    # model.apply(initWeights)
    print("Generating network costs %s seconds."%(time.time()-startTime))

    preOptimizer = torch.optim.Adam(model.parameters(),lr=params["preLr"])
    optimizer = torch.optim.Adam(model.parameters(),lr=params["lr"],weight_decay=params["decay"])
    # scheduler = StepLR(optimizer,step_size=params["step_size"],gamma=params["gamma"])
    scheduler = MultiStepLR(optimizer,milestones=params["milestone"],gamma=params["gamma"])
    # schedulerFun = lambda epoch: ((epoch+100)/(epoch+101))
    # scheduler = MultiplicativeLR(optimizer,lr_lambda=schedulerFun)

    startTime = time.time()
    preTrain(model,device,params,preOptimizer,None,rough)
    train(model,device,params,optimizer,scheduler)
    print("Training costs %s seconds."%(time.time()-startTime))

    model.eval()
    testError = test(model,device,params)
    print("The test error (of the last model) is %s."%testError)
    print("The number of parameters is %s,"%count_parameters(model))

    torch.save(model.state_dict(),"last_model.pt")

if __name__=="__main__":
    main()