# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 18:25:12 2023

@author: ashis
"""


# Coupled nODE training with changeable training length 
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from torchdiffeq import odeint

# Loss function

def LossConst(recon, data):
        
    # add gamma
    return (torch.norm(recon-data)) / nDim

# NN-architecture (Encode-decoder)
class Encoder(nn.Module):
    
    def __init__(self, encoded_space_dim,input_dim):
        super().__init__()
        
        
        # Convolutional blocks
        self.encoder_cnn_1 = nn.Sequential(
            nn.Conv1d(1, 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(2),
            nn.Sigmoid(),
            nn.Conv1d(2, 2, kernel_size=3, stride=1, padding=1))
        
        self.encoder_cnn_2 = nn.Sequential(
            nn.Conv1d(2, 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(4),
            nn.Sigmoid(),
            nn.Conv1d(4, 4, kernel_size=3, stride=1, padding=1))

        self.encoder_cnn_3 = nn.Sequential(
            nn.Conv1d(4, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(8),
            nn.Sigmoid(),
            nn.Conv1d(8, 8, kernel_size=3, stride=1, padding=1))

        self.encoder_cnn_4 = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.Sigmoid(),
            nn.Conv1d(16, 16, kernel_size=3, stride=1, padding=1))

        
        # Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        
        # Linear section
        self.encoder_lin = nn.Linear(input_dim, encoded_space_dim)

        
    def forward(self, x):
        
        # Convolutional blocks
        x = self.encoder_cnn_1(x)
        x = self.encoder_cnn_2(x)
        x = self.encoder_cnn_3(x)
        x = self.encoder_cnn_4(x)
        
        # Flatten layer
        x = self.flatten(x)
        
        # Fully connected-layer
        x = self.encoder_lin(x)
        return x
    
class Decoder(nn.Module):
    
    def __init__(self, encoded_space_dim,input_dim):
        super().__init__()
        
        # Fully-connected layer
        self.decoder_lin =  nn.Linear(encoded_space_dim, input_dim)

        # Unflatten layer
        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(16, int(input_dim/16)))

        # Deconvolution blocks
        self.decoder_cnn_4 = nn.Sequential(
            nn.ConvTranspose1d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
            nn.BatchNorm1d(16),
            nn.ConvTranspose1d(16, 8, kernel_size=4, stride=2, padding=1))

        self.decoder_cnn_3 = nn.Sequential(
            nn.ConvTranspose1d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
            nn.BatchNorm1d(8),
            nn.ConvTranspose1d(8, 4, kernel_size=4, stride=2, padding=1))

        self.decoder_cnn_2 = nn.Sequential(
            nn.ConvTranspose1d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
            nn.BatchNorm1d(4),
            nn.ConvTranspose1d(4, 2, kernel_size=4, stride=2, padding=1))

        self.decoder_cnn_1 = nn.Sequential(
            nn.ConvTranspose1d(2, 2, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
            nn.BatchNorm1d(2),
            nn.ConvTranspose1d(2, 1, kernel_size=4, stride=2, padding=1))


        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_cnn_4(x)
        x = self.decoder_cnn_3(x)
        x = self.decoder_cnn_2(x)
        x = self.decoder_cnn_1(x)
        #x = torch.sigmoid(x)
        return x


# NN-architecture (for n-ODE)
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

gamma = 1e1
dt = 0.001953125

# File Paths
TrainDataPath = '../Datasets/Initial_cond/Full Space/dataComb.npy'

# Training params
trainParams = [1,2,3,4,5,6]
sampSize   = 5000 # Size of each sample

# Testing params
testParams = [7,8,9]

# Trajectory length
trajLen = 1
trajLen += 1


# Loading latent space training data 
TrainData = np.load(TrainDataPath)
nDim = TrainData.shape[1]
TrainData = torch.tensor(TrainData, dtype=torch.float32)

for i in range(len(trainParams)):
    # Extracting current sample
    currSamp = TrainData[i*sampSize:(i+1)*sampSize,:]
    
    # Removing transient part
    currSamp = currSamp[1000:, :]
    
    # sample shape
    sampShape = currSamp.shape
    
    # Output array
    currOut = torch.zeros((trajLen,sampShape[0],sampShape[1]))
    for j in range(trajLen):
        currOut[j,:,:] = currSamp[j:trajLen:(trajLen-j)]
    
    # Input array
    currInp = torch.zeros((1,sampShape[0],sampShape[1]))
    currInp[0,:,:] = currOut[0,:,:]
    
    if i == 0:
        inputArr = currInp
        outputArr = currOut
        
    else:
        inputArr = torch.concatenate((inputArr, currInp), axis=1)
        outputArr = torch.concatenate((outputArr, currOut), axis=1)
        

# Defining nODE-model
latDim = 10
model = FullyConnectedNet(latDim, latDim, 100, 5)

# Defining encoder-decoder
d = 10
encoder = Encoder(d,nDim)
decoder = Decoder(d,nDim)

params_to_optimize = [
    {'params': encoder.parameters()},
    {'params': decoder.parameters()},
    {'params': model.parameters()}
]


##### Initializing Loss and optimizer ########
# Loss
lossFunc = nn.MSELoss()

# Learning Rate
lr= 0.001


# Set the random seed for reproducible results
torch.manual_seed(0)


optimizer = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)



##### Training loop ###########
nEpochs = int(0.5e4)

trainingLoss = []
model.train()
t_batch = dt * (torch.linspace(0,trajLen,trajLen+1)[:trajLen])

for i in range(nEpochs):
    
    optimizer.zero_grad()
    y0 = inputArr
    
    # Reshaping input
    y0 = y0.squeeze()
    y0 = y0[:, None, :]
    
    # Encoding to latent space 
    y0Lat = encoder(y0)
    
    # Reshaping
    y0Lat = y0Lat[None, :, :]
    
    # Learning latent dynamics
    pred_y = odeint(model, y0Lat, t_batch, method='euler')
    pred_y = pred_y.squeeze()
    
    # Decoding to full space
    pred_y_full = torch.zeros((trajLen, outputArr.shape[1], nDim))
    for i in range(trajLen):
        pred_y_full[i,:,:] = decoder(pred_y[i,:,:]).squeeze()
        
    
    loss = LossConst(pred_y_full, outputArr)
    
    loss.backward()
    
    optimizer.step()
    
    # Printing Loss
    print('\t Train loss : {}'.format(loss.data))
    trainingLoss.append(loss.detach().cpu().numpy())
    
    
# Saving model 
torch.save(model, 'latModelnODE_{}_varLen_{}.pt'.format(gamma, trajLen-1))

# Plotting loss
plt.loglog(trainingLoss)
plt.xlabel('Iterations')
plt.ylabel('Loss ($L_2$ norm)')
plt.savefig('ConvergenceLatModel_{}.png'.format(gamma))
