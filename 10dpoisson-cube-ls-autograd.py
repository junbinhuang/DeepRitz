import numpy as np 
import math, torch, generateData, time
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, StepLR
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
        # x = torch.tanh(self.linearIn(x)) # Match dimension
        for i in range(len(self.linear)//2):
            x_temp = torch.tanh(self.linear[2*i](x))
            x_temp = torch.tanh(self.linear[2*i+1](x_temp))
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

    data1 = torch.rand(params["bodyBatch"],params["d"]).float().to(device)
    data1.requires_grad = True
    data2 = torch.rand(2*params["d"]*(params["bdryBatch"]//(2*params["d"])),params["d"]).float().to(device)
    temp = params["bdryBatch"]//(2*params["d"])
    for i in range(params["d"]):
        data2[(2*i+0)*temp:(2*i+1)*temp,i] = 0.0
        data2[(2*i+1)*temp:(2*i+2)*temp,i] = 1.0

    for step in range(params["trainStep"]-params["preStep"]):
        output1 = model(data1)

        model.zero_grad()

        dfdx = torch.autograd.grad(output1,data1,grad_outputs=torch.ones_like(output1),retain_graph=True,create_graph=True,only_inputs=True)[0]
        dfdx20 = torch.autograd.grad(dfdx[:,0].unsqueeze(1),data1,grad_outputs=torch.ones_like(output1),retain_graph=True,create_graph=True,only_inputs=True)[0][:,0].unsqueeze(1)
        dfdx21 = torch.autograd.grad(dfdx[:,1].unsqueeze(1),data1,grad_outputs=torch.ones_like(output1),retain_graph=True,create_graph=True,only_inputs=True)[0][:,1].unsqueeze(1)
        dfdx22 = torch.autograd.grad(dfdx[:,2].unsqueeze(1),data1,grad_outputs=torch.ones_like(output1),retain_graph=True,create_graph=True,only_inputs=True)[0][:,2].unsqueeze(1)
        dfdx23 = torch.autograd.grad(dfdx[:,3].unsqueeze(1),data1,grad_outputs=torch.ones_like(output1),retain_graph=True,create_graph=True,only_inputs=True)[0][:,3].unsqueeze(1)
        dfdx24 = torch.autograd.grad(dfdx[:,4].unsqueeze(1),data1,grad_outputs=torch.ones_like(output1),retain_graph=True,create_graph=True,only_inputs=True)[0][:,4].unsqueeze(1)
        dfdx25 = torch.autograd.grad(dfdx[:,5].unsqueeze(1),data1,grad_outputs=torch.ones_like(output1),retain_graph=True,create_graph=True,only_inputs=True)[0][:,5].unsqueeze(1)
        dfdx26 = torch.autograd.grad(dfdx[:,6].unsqueeze(1),data1,grad_outputs=torch.ones_like(output1),retain_graph=True,create_graph=True,only_inputs=True)[0][:,6].unsqueeze(1)
        dfdx27 = torch.autograd.grad(dfdx[:,7].unsqueeze(1),data1,grad_outputs=torch.ones_like(output1),retain_graph=True,create_graph=True,only_inputs=True)[0][:,7].unsqueeze(1)
        dfdx28 = torch.autograd.grad(dfdx[:,8].unsqueeze(1),data1,grad_outputs=torch.ones_like(output1),retain_graph=True,create_graph=True,only_inputs=True)[0][:,8].unsqueeze(1)
        dfdx29 = torch.autograd.grad(dfdx[:,9].unsqueeze(1),data1,grad_outputs=torch.ones_like(output1),retain_graph=True,create_graph=True,only_inputs=True)[0][:,9].unsqueeze(1)
        # Loss function 1
        fTerm = ffun(data1).to(device)
        loss1 = torch.mean((dfdx20+dfdx21+dfdx22+dfdx23+dfdx24+dfdx25+dfdx26+dfdx27+dfdx28+dfdx29+fTerm)*\
            (dfdx20+dfdx21+dfdx22+dfdx23+dfdx24+dfdx25+dfdx26+dfdx27+dfdx28+dfdx29+fTerm))

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
            data1 = torch.rand(params["bodyBatch"],params["d"]).float().to(device)
            data1.requires_grad = True
            data2 = torch.rand(2*params["d"]*(params["bdryBatch"]//(2*params["d"])),params["d"]).float().to(device)
            temp = params["bdryBatch"]//(2*params["d"])
            for i in range(params["d"]):
                data2[(2*i+0)*temp:(2*i+1)*temp,i] = 0.0
                data2[(2*i+1)*temp:(2*i+2)*temp,i] = 1.0

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

    data = torch.rand(numQuad,10).float().to(device)
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
    # torch.manual_seed(21)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    params = dict()
    params["radius"] = 1
    params["d"] = 10 # 10D
    params["dd"] = 1 # Scalar field
    params["bodyBatch"] = 1024 # Batch size
    params["bdryBatch"] = 2000 # Batch size for the boundary integral
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
    params["area"] = 20
    params["step_size"] = 5000
    params["milestone"] = [5000,10000,20000,35000,48000]
    params["gamma"] = 0.5
    params["decay"] = 0.00001

    startTime = time.time()
    model = RitzNet(params).to(device)
    model.apply(initWeights)
    print("Generating network costs %s seconds."%(time.time()-startTime))

    # torch.seed()
    preOptimizer = torch.optim.Adam(model.parameters(),lr=params["preLr"])
    optimizer = torch.optim.Adam(model.parameters(),lr=params["lr"],weight_decay=params["decay"])
    # scheduler = StepLR(optimizer,step_size=params["step_size"],gamma=params["gamma"])
    scheduler = MultiStepLR(optimizer,milestones=params["milestone"],gamma=params["gamma"])

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