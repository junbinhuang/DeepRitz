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
        self.linearIn = nn.Linear(self.params["d"], self.params["width"])
        self.linear = nn.ModuleList()
        for _ in range(params["depth"]):
            self.linear.append(nn.Linear(self.params["width"], self.params["width"]))

        self.linearOut = nn.Linear(self.params["width"], self.params["dd"])

    def forward(self, x):
        x = F.softplus(self.linearIn(x)) # Match dimension
        for layer in self.linear: # Is ResNets really needed?
            x_temp = F.softplus(layer(x))
            x = x_temp+x
        # for i in range(len(self.linear)//2): # Use the network structure proposed in the paper.
        #     x_temp = F.softplus(self.linear[2*i](x))
        #     x_temp = F.softplus(self.linear[2*i+1](x_temp))
        #     x = x_temp+x
        
        return self.linearOut(x)

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
    data2 = torch.from_numpy(generateData.sampleFromSurface10(params["radius"],params["bdryBatch"])).float().to(device)
    x_shift = torch.from_numpy(np.eye(10)*params["diff"]).float().to(device)
    data1_shift0 = data1+x_shift[0]
    data1_shift1 = data1+x_shift[1]
    data1_shift2 = data1+x_shift[2]
    data1_shift3 = data1+x_shift[3]
    data1_shift4 = data1+x_shift[4]
    data1_shift5 = data1+x_shift[5]
    data1_shift6 = data1+x_shift[6]
    data1_shift7 = data1+x_shift[7]
    data1_shift8 = data1+x_shift[8]
    data1_shift9 = data1+x_shift[9]

    for step in range(params["trainStep"]-params["preStep"]):
        output1 = model(data1)
        output1_shift0 = model(data1_shift0)
        output1_shift1 = model(data1_shift1)
        output1_shift2 = model(data1_shift2)
        output1_shift3 = model(data1_shift3)
        output1_shift4 = model(data1_shift4)
        output1_shift5 = model(data1_shift5)
        output1_shift6 = model(data1_shift6)
        output1_shift7 = model(data1_shift7)
        output1_shift8 = model(data1_shift8)
        output1_shift9 = model(data1_shift9)

        dfdx0 = (output1_shift0-output1)/params["diff"] # Use difference to approximate derivatives.
        dfdx1 = (output1_shift1-output1)/params["diff"] # The PyTorch autograd is not very effective in this case.
        dfdx2 = (output1_shift2-output1)/params["diff"]
        dfdx3 = (output1_shift3-output1)/params["diff"]
        dfdx4 = (output1_shift4-output1)/params["diff"]
        dfdx5 = (output1_shift5-output1)/params["diff"]
        dfdx6 = (output1_shift6-output1)/params["diff"]
        dfdx7 = (output1_shift7-output1)/params["diff"]
        dfdx8 = (output1_shift8-output1)/params["diff"]
        dfdx9 = (output1_shift9-output1)/params["diff"]

        model.zero_grad()

        # Loss function 1
        fTerm = ffun(data1).to(device)
        loss1 = torch.mean(0.5*(dfdx0*dfdx0 + dfdx1*dfdx1 + dfdx2*dfdx2 +\
            dfdx3*dfdx3 + dfdx4*dfdx4 + dfdx5*dfdx5 + dfdx6*dfdx6 +\
                dfdx7*dfdx7 + dfdx8*dfdx8 + dfdx9*dfdx9)-fTerm*output1)

        # Loss function 2
        output2 = model(data2)
        target2 = exact(params["radius"],data2)
        loss2 = torch.mean((output2-target2)*(output2-target2) * params["penalty"] *params["area"])
        loss = loss1+loss2

        # if step == 0:
        #     torch.save(model.state_dict(),"best_model.pt")
        #     last_loss = loss.item()
        
        # if loss.item()<last_loss-0.01*abs(last_loss):
        #     torch.save(model.state_dict(),"best_model.pt")
        #     last_loss = loss.item()                    

        if step%params["writeStep"] == params["writeStep"]-1:
            with torch.no_grad():
                target = exact(params["radius"],data1)
                error = errorFun(output1,target,params)
                print("Error at Step %s is %s."%(step+params["preStep"]+1,error))
                # params["penalty"] *= 1.01
            file = open("lossData.txt","a")
            file.write(str(step+params["preStep"]+1)+" "+str(error)+"\n")

        data1 = torch.from_numpy(generateData.sampleFromDisk10(params["radius"],params["bodyBatch"])).float().to(device)
        data2 = torch.from_numpy(generateData.sampleFromSurface10(params["radius"],params["bdryBatch"])).float().to(device)

        data1_shift0 = data1+x_shift[0]
        data1_shift1 = data1+x_shift[1]
        data1_shift2 = data1+x_shift[2]
        data1_shift3 = data1+x_shift[3]
        data1_shift4 = data1+x_shift[4]
        data1_shift5 = data1+x_shift[5]
        data1_shift6 = data1+x_shift[6]
        data1_shift7 = data1+x_shift[7]
        data1_shift8 = data1+x_shift[8]
        data1_shift9 = data1+x_shift[9]

        loss.backward()

        # Update the weights.
        # if step == 10000: 
        #     params["bdryBatch"] *= 2
        #     params["bodyBatch"] *= 2
        # if step == 20000: 
        #     params["bdryBatch"] *= 2
        #     params["bodyBatch"] *= 2

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
    params["bodyBatch"] = 128 # Batch size
    params["bdryBatch"] = 128 # Batch size for the boundary integral
    params["lr"] = 0.01 # Learning rate
    params["preLr"] = 0.01 # Learning rate (Pre-training)
    params["width"] = 20 # Width of layers
    params["depth"] = 6 # Depth of the network: depth+2
    params["numQuad"] = 40000 # Number of quadrature points for testing
    params["trainStep"] = 50000
    params["penalty"] = 500
    params["preStep"] = 0
    params["diff"] = 0.0001
    params["writeStep"] = 50
    params["area"] = areaVolume(params["radius"],params["d"])
    params["step_size"] = 5000
    params["gamma"] = 0.3

    startTime = time.time()
    model = RitzNet(params).to(device)
    print("Generating network costs %s seconds."%(time.time()-startTime))

    preOptimizer = torch.optim.Adam(model.parameters(),lr=params["preLr"])
    optimizer = torch.optim.Adam(model.parameters(),lr=params["lr"])
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

    # model = RitzNet(params).to(device)
    # model.load_state_dict(torch.load("last_model.pt"))
    # model.eval()

    # testError = test(model,device,params)
    # print("The test error (of the saved model) is %s."%testError)

if __name__=="__main__":
    main()