# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 22:08:55 2023

@author: ashis
"""

# nODE training with changeable training length 
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from torchdiffeq import odeint
import time


# Loss function

def LossConst(recon, data):
        
    loss = torch.nn.MSELoss()
    # add gamma
    return loss(recon, data)



# NN-architecture
class FullyConnectedNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(FullyConnectedNet, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Create hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.hidden_layers.append(nn.Linear(input_size, hidden_size))
            else:
                self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        # ELU activation
        self.activation = nn.ELU()
    
    def forward(self, t, y):
        # Pass input through hidden layers
        for i in range(self.num_layers):
            y = self.hidden_layers[i](y)
            y = self.activation(y)
        
        # Pass through output layer
        y = self.output_layer(y)
        
        return y

# Trajectory length
#trajLenParams = [1, 3, 7, 9, 19, 49, 99, 199, 499, 999, 1999]
trajLenParams = [999]
finalLoss     = []

gamma = 1e1
dt = 0.001953125

# File Paths
modelPath = '../Models/Initial_cond/Lat Space nODE/'.format(gamma)
LatTrainDataPath = '../Datasets/dataComb_lat.npy'.format(gamma)
LatTestDataPath  = '../Datasets/Initial_cond/Latent Space/Gamma={}/'.format(gamma)
resultsPath   = '../Results/K-S Equations/Initial_cond/Lat Space nODE/'

# Training params
trainParams = [1,2,3,4,5,6]
sampSize   = 5000 # Size of each sample

# Testing params
testParams = [7,8,9]

# Loading latent space training data 
LatTrainData = np.load(LatTrainDataPath)
latDim = LatTrainData.shape[1]
LatTrainData = torch.tensor(LatTrainData, dtype=torch.float32)


for trajLen in trajLenParams:

    
    # Trajectory length
    #trajLen = 1
    trajLen += 1
    
    
    
    for i in range(len(trainParams)):
        # Extracting current sample
        currSamp = LatTrainData[i*sampSize:(i+1)*sampSize,:]
        
        # Removing transient part
        currSamp = currSamp[1000:, :]
        
        # sample shape
        sampShape = currSamp.shape
        
        # Output array
        currOut = torch.zeros((trajLen,int(sampShape[0]/trajLen),sampShape[1]))
        for j in range(int(sampShape[0]/trajLen)):
            currOut[:,j,:] = currSamp[j*trajLen:(j+1)*trajLen, :]
        
        # Input array
        currInp = torch.zeros((1,int(sampShape[0]/trajLen),sampShape[1]))
        currInp[0,:,:] = currOut[0,:,:]
        
        if i == 0:
            inputArr = currInp
            outputArr = currOut
            
        else:
            inputArr = torch.concatenate((inputArr, currInp), axis=1)
            outputArr = torch.concatenate((outputArr, currOut), axis=1)
            
    
    # Defining model
    model = FullyConnectedNet(latDim, latDim, 100, 5)
    
    
    
    ##### Initializing Loss and optimizer ########
    # Loss
    lossFunc = nn.MSELoss()
    
    # Learning Rate
    lr= 0.001
    
    
    # Set the random seed for reproducible results
    torch.manual_seed(0)
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-05)
    
    
    
    ##### Training loop ###########
    nEpochs = int(0.5e4) #0.5e4
    
    trainingLoss = []
    model.train()
    t_batch = dt * (torch.linspace(0,trajLen,trajLen+1)[:trajLen])

    time1 = time.time()    
    for i in range(nEpochs):
        
        optimizer.zero_grad()
        y0 = inputArr
        
        pred_y = odeint(model, y0, t_batch, method='euler')
        
        loss = LossConst(pred_y.squeeze(), outputArr)
        
        loss.backward()
        
        optimizer.step()
        
        # Printing Loss
        if i%100==0:
          time2 = time.time()
          print('\t{}. Train loss : {} \t time:{}'.format(i,loss.data,(time2-time1)))
          time1 = time2
        
        trainingLoss.append(loss.detach().cpu().numpy())
        
        
    # Saving model 
    torch.save(model, 'latModelnODE_{}_varLen_{}.pt'.format(gamma, trajLen-1))
    
    # Plotting loss
    plt.loglog(trainingLoss)
    plt.xlabel('Iterations')
    plt.ylabel('Loss ($L_2$ norm)')
    plt.savefig('ConvergenceLatModel_{}_varLen_{}.png'.format(gamma, trajLen-1))
    
    # Appending final loss
    finalLoss.append(trainingLoss[-1])
    
    # deleting everything
    del model, optimizer, trainingLoss, inputArr, outputArr


# Plotting loss
plt.loglog(trajLenParams, finalLoss)
plt.xlabel('Trajectory length')
plt.ylabel('Reconstruction Loss ($L_2$ norm)')
plt.savefig('LossvsLen.png')


