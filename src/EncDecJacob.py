# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 20:54:46 2023

@author: ashis
"""

# Encoder and decoder jacobian 

import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np



# NN-architecture
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


gamma = 10.0

# File Paths
modelPath = '../Models/Gamma={}/'.format(gamma)
trainDataPath = '../Datasets/dataComb.npy'


# Loading Model
encoder = torch.load(modelPath+'encoder_{}.pt'.format(gamma))
encoder.eval()

decoder = torch.load(modelPath+'decoder_{}.pt'.format(gamma))
decoder.eval()

# Loading Training Data 
trainData = np.load(trainDataPath)
nDim = trainData.shape[1]
trainData = torch.tensor(trainData, dtype=torch.float32)
trainData = trainData[:,None,:]


# Extracting one sample
sample = trainData[1000,:,:]
sample = sample[None,:,:]


# Computing encoder jacobian
encJac = torch.autograd.functional.jacobian(encoder, sample)
encJac = encJac.squeeze()

# Computing decoder jacobian
decJac = torch.autograd.functional.jacobian(decoder, encoder(sample))
decJac = decJac.squeeze()

prod = decJac @ encJac
prod = prod.detach().numpy()

plt.imshow(prod, cmap='hot', interpolation='nearest')
plt.show()



