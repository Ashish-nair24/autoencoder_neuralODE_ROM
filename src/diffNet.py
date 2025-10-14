# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 17:05:51 2023

@author: ashis
"""

# Training diff network
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np


# Loss function

def LossConst(recon, data):
        
    # add gamma
    return (torch.norm(recon-data)) / latDim


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
    
    def forward(self, x):
        # Pass input through hidden layers
        for i in range(self.num_layers):
            x = self.hidden_layers[i](x)
            x = self.activation(x)
        
        # Pass through output layer
        x = self.output_layer(x)
        
        return x



# Global parameters
gamma = 1e1
dt = 0.001953125

# File Paths
LatTrainDataPath = '../Datasets/Initial_cond/Latent Space/Gamma={}/dataComb_lat.npy'.format(gamma)
LatTestDataPath  = '../Datasets/Initial_cond/Full Space/'
resultsPath   = '../Results/K-S Equations/Initial_cond/'


# Loading latent space training data 
LatTrainData = np.load(LatTrainDataPath)
latDim = LatTrainData.shape[1]
LatTrainData = torch.tensor(LatTrainData, dtype=torch.float32)

# Stacking training and target data
nTrainSamp = 6 # Number of training samples
sampSize   = 5000 # Size of each sample

for i in range(nTrainSamp):
    
    if i==0:
        trainData = LatTrainData[0:(sampSize-1), :]
        targetData = (LatTrainData[1:sampSize, :] - LatTrainData[0:(sampSize-1), :])/dt
    else:
        trainData = torch.concatenate((trainData, LatTrainData[i*sampSize:((i+1)*sampSize-1), :]))
        targetData = torch.concatenate((targetData, ((LatTrainData[i*sampSize+1:((i+1)*sampSize), :]-LatTrainData[i*sampSize:((i+1)*sampSize-1), :])/dt)))

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
nEpochs = int(0.5e4)

trainingLoss = []

model.train()


for i in range(nEpochs):
    
    
    # Perform reconstruction
    recon = model(trainData)    
    
    # Evaluate loss
    # loss = lossFunc(recon, Data)
    loss = LossConst(recon, targetData)
    
    # Backpropogating gradients
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Printing Loss
    print('\t Train loss : {}'.format(loss.data))
    trainingLoss.append(loss.detach().cpu().numpy())
    

# Saving model 
torch.save(model, 'latModel_{}.pt'.format(gamma))

# Plotting loss
plt.loglog(trainingLoss)
plt.xlabel('Iterations')
plt.ylabel('Loss ($L_2$ norm)')
plt.savefig('ConvergenceLatModel_{}.png'.format(gamma))








