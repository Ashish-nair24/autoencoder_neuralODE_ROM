# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 15:03:39 2023

@author: ashis
"""

# Sample convolutional NN for K-S dataset 


import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np


# Loss function

def LossConst(recon, data, lat, latTilde):
    
    gamma = 1
    
    # add gamma
    return (torch.norm(recon-data) + gamma * torch.norm(lat-latTilde)) / nDim


# NN-architecture
#class Encoder(nn.Module):
    
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
    
#class Decoder(nn.Module):
    
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
    
class Encoder(nn.Module):
    
    def __init__(self, encoded_space_dim,input_dim,num_layers, cnn=True):
        super().__init__()
        
        self.cnn = cnn
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.encoded_space_dim = encoded_space_dim


        if not cnn:
            #print('fnn')
            self.encoder_fnn_1 = nn.Sequential(nn.Linear(input_dim, int(input_dim/2)), nn.ELU())
            self.encoder_fnn_2 = nn.Sequential(nn.Linear(int(input_dim/2), int(input_dim/4)), nn.ELU())
            self.encoder_fnn_3 = nn.Sequential(nn.Linear(int(input_dim/4), int(input_dim/8)), nn.ELU())
            self.encoder_fnn_4 = nn.Sequential(nn.Linear(int(input_dim/8), int(input_dim/16)), nn.ELU())
            self.encoder_fnn_5 = nn.Sequential(nn.Linear(int(input_dim/16), int(input_dim/32)), nn.ELU())
            self.encoder_fnn_6 = nn.Sequential(nn.Linear(int(input_dim/32), int(encoded_space_dim)), nn.ELU())
        
    def forward(self, x):
        
        if self.cnn:
            currInp = int(1)
            currOut = int(currInp * 2)
            for i in range(self.num_layers):
                x =  nn.Conv1d(currInp, currOut, kernel_size=4, stride=2, padding=1) (x)
                x =  nn.ELU()(x)
                x =  nn.Conv1d(currOut, currOut, kernel_size=3, stride=1, padding=1)(x)
                currInp = currOut
                currOut = int(currInp * 2)

            
            # Flatten layer
            x = nn.Flatten(start_dim=1)(x)
            
            # Fully connected-layer
            x = nn.Linear(self.input_dim, self.encoded_space_dim)(x)
        
        else:


            x = x.squeeze()
            # FNN Blocks
            x = self.encoder_fnn_1(x)
            x = self.encoder_fnn_2(x)
            x = self.encoder_fnn_3(x)
            x = self.encoder_fnn_4(x)
            x = self.encoder_fnn_5(x)
            x = self.encoder_fnn_6(x)


        return x
    
class Decoder(nn.Module):
    
    def __init__(self, encoded_space_dim, input_dim, num_layers, cnn=True):
        super().__init__()
        
        self.cnn = cnn
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.encoded_space_dim = encoded_space_dim
        

        if not cnn:
            #print('fnn')
            self.decoder_fnn_1 = nn.Sequential(nn.Linear(int(input_dim/2), input_dim), nn.ELU())
            self.decoder_fnn_2 = nn.Sequential(nn.Linear(int(input_dim/4), int(input_dim/2)), nn.ELU())
            self.decoder_fnn_3 = nn.Sequential(nn.Linear(int(input_dim/8), int(input_dim/4)), nn.ELU())
            self.decoder_fnn_4 = nn.Sequential(nn.Linear( int(input_dim/16), int(input_dim/8)), nn.ELU())
            self.decoder_fnn_5 = nn.Sequential(nn.Linear( int(input_dim/32), int(input_dim/16)), nn.ELU())
            self.decoder_fnn_6 = nn.Sequential(nn.Linear( int(encoded_space_dim), int(input_dim/32)), nn.ELU())

        
    def forward(self, x):

        if self.cnn:

            # Linear Layer
            x = nn.Linear(self.encoded_space_dim, self.input_dim)(x)

            # Unflatten Layer
            x = nn.Unflatten(dim=1, unflattened_size=(int(2**self.num_layers), int(self.input_dim/(2**self.num_layers))))(x)

            # deconvolution layers
            currInp = int(2**self.num_layers)
            currOut = int(currInp / 2)

            for i in range(self.num_layers):
                x = nn.ConvTranspose1d(currInp, currInp, kernel_size=3, stride=1, padding=1)(x)
                x = nn.ELU()(x)
                x = nn.ConvTranspose1d(currInp, currOut, kernel_size=4, stride=2, padding=1)(x)
                currInp = currOut
                currOut = int(currInp / 2)




        else:

            # FNN Blocks
            x = self.decoder_fnn_6(x)
            x = self.decoder_fnn_5(x)
            x = self.decoder_fnn_4(x)
            x = self.decoder_fnn_3(x)
            x = self.decoder_fnn_2(x)
            x = self.decoder_fnn_1(x)
            x = x.unsqueeze(1)

        return x
    

# test = torch.randn((500,1,128))


# # Sanity check
# Enc = Encoder(10,128)
# Dec = Decoder(10,128)

# lat = Enc(test)
# fun = Dec(lat)

###### Data #############
# Loading Data-set 
# dataFilePath = '../Datasets/KS_matlab.mat'

# Data = sio.loadmat(dataFilePath)
# Data = Data['uu']

Data = np.load('../Datasets/Initial_cond/Full Space/dataSet_1.npy')

# Pre-processing data 
# Transpose (samples x dims)
#Data = Data.T
nDim = Data.shape[1]

# Converting to tensor
Data = torch.tensor(Data, dtype=torch.float32)

# Adding channel dimension
Data = Data[:,None,:]


# Sanity check on data 
# #Data1 = torch.rand((500,1,128))
# Enc = Encoder(10,128)
# Dec = Decoder(10,128)

# lat = Enc(Data)
# recon = Dec(lat)


##### Initializing Loss and optimizer ########
# Loss
lossFunc = nn.MSELoss()

# Learning Rate
lr= 0.001

# Set the random seed for reproducible results
torch.manual_seed(0)

# Initialize the two networks
d = 10
encoder = Encoder(d,nDim,6,True)
decoder = Decoder(d,nDim,6,True)
params_to_optimize = [
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
]

optimizer = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)



##### Training loop ###########
nEpochs = int(100)

trainingLoss = []

encoder.train()
decoder.train()


for i in range(nEpochs):
    
    
    # Perform reconstruction
    lat   = encoder(Data)
    recon = decoder(lat)
    
    latTilde = encoder(decoder(lat))
    
    
    # Evaluate loss
    # loss = lossFunc(recon, Data)
    loss = LossConst(recon, Data, lat, latTilde)
    
    # Backpropogating gradients
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Printing Loss
    print('\t Train loss : {}'.format(loss.data))
    trainingLoss.append(loss.detach().cpu().numpy())
    



# Saving model 
torch.save(encoder, 'encoder.pt')
torch.save(decoder, 'decoder.pt')
    
# Plotting loss
# plt.plot(trainingLoss)
# plt.xlabel('Iterations')
# plt.ylabel('Loss ($L_2$ norm)')
# plt.show()

# # Reconstruction
# lat   = encoder(Data)  
# recon = decoder(lat)


# # Reshaping 
# recon = torch.squeeze(recon)


# # Plotting 
# fig, ax = plt.subplots()
# CS = ax.contourf(recon.detach().numpy(), levels=500, colormap='coolwarm')
# ax.set_title('Reconstruction')
# ax.set_xlabel('x')
# ax.set_ylabel('t')
# cbar = fig.colorbar(CS)
# plt.savefig('reconstruction.png')
# plt.close()


# Data = torch.squeeze(Data)
# fig, ax = plt.subplots()
# CS = ax.contourf(Data.detach().numpy(), levels=500, colormap='coolwarm')
# ax.set_title('Data')
# ax.set_xlabel('x')
# ax.set_ylabel('t')
# cbar = fig.colorbar(CS)
# plt.savefig('Data.png')
# plt.close()





