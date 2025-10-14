# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 22:32:49 2023

@author: ashis
"""

# Inference tests 
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import copy 
import os 
import torch.distributed as dist
# Loss function

def LossConst(recon, data, lat, latTilde):
    
    loss = nn.MSELoss()
    gamma = 0

    # add gamma
    return (loss(recon, data) + gamma * loss(lat, latTilde))


def reconLoss(recon, data):
    
    loss = nn.MSELoss()
    
    return (loss(recon, data))


# NN- Architecture
class Encoder(nn.Module):
    
    def __init__(self, encoded_space_dim,input_dim,num_layers, cnn=True):
        super().__init__()
        
        self.cnn = cnn
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.encoded_space_dim = encoded_space_dim

        if cnn:
            encoder_layers=[]
            currInp = int(1)
            currOut = int(currInp * 2)
            for i in range(self.num_layers):
                encoder_layers.extend([nn.Conv1d(currInp, currOut, kernel_size=4, stride=2, padding=1),
                                       nn.ELU(),
                                       nn.Conv1d(currOut, currOut, kernel_size=3, stride=1, padding=1),])
                
                currInp = currOut
                currOut = int(currInp * 2)

            self.encoder_cnn_layers = nn.Sequential(*encoder_layers)
            self.encoder_flatten = nn.Flatten(start_dim=1)
            self.encoder_linear  = nn.Linear(self.input_dim, self.encoded_space_dim)
        
        
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
            x = self.encoder_cnn_layers(x)
            x = self.encoder_flatten(x)
            x = self.encoder_linear(x)

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
        
        if cnn:
            decoder_layers=[]
            currInp = int(2**self.num_layers)
            currOut = int(currInp / 2)
            for i in range(self.num_layers):
                decoder_layers.extend([nn.ConvTranspose1d(currInp, currInp, kernel_size=3, stride=1, padding=1),
                                       nn.ELU(),
                                       nn.ConvTranspose1d(currInp, currOut, kernel_size=4, stride=2, padding=1),])
                
                currInp = currOut
                currOut = int(currInp / 2)

            self.decoder_cnn_layers = nn.Sequential(*decoder_layers)
            self.decoder_unflatten = nn.Unflatten(dim=1, unflattened_size=(int(2**self.num_layers), int(self.input_dim/(2**self.num_layers))))
            self.decoder_linear  = nn.Linear(self.encoded_space_dim, self.input_dim)

        
        
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

            x = self.decoder_linear(x)
            x = self.decoder_unflatten(x)
            x = self.decoder_cnn_layers(x)



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


gamma = 0.0
numLayers = 8

# File Paths
modelPath = '../Models/Initial_cond/numLayers_Test/'#.format(gamma)
trainDataPath = '../Datasets/Initial_cond/Full Space/dataComb_InitCond.npy'
testDataPath  = '../Datasets/Initial_cond/Full Space/'
resultsPath   = '../Results/K-S Equations/Initial_cond/numLayers_Test/'
 

# Save latent variables for Node training
saveLatent = True

if saveLatent:
    latentPath = '../Datasets/Initial_cond/Latent Space/numLayers_Test/'#.format(gamma)

# Parameter space (Initial Conditions)
testParams = [23,24,25]

# Initialising environment variables
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
dist.init_process_group('gloo')


# Loading Model
# encoder = torch.load(modelPath+'encoder_{}.pt'.format(gamma), map_location=torch.device('cpu'))
encoder = torch.load(modelPath+'encoder_{}_nLayers_{}.pt'.format(gamma, numLayers), map_location=torch.device('cpu'))
encoder.eval()

encoder_dict = encoder.state_dict()
encoder_dict_copy = copy.deepcopy(encoder_dict)
del encoder

encoder = Encoder(10, 512, numLayers, True)
for k in encoder_dict.keys():
    k_old = k
    if len(k.split('module.')) == 2:
        k_new = k.split('module.')[1]
    else:
        k_new = k.split('module.')[0]
    encoder_dict_copy[k_new] = encoder_dict_copy.pop(k_old)

encoder.load_state_dict(encoder_dict_copy)

# decoder = torch.load(modelPath+'decoder_{}.pt'.format(gamma), map_location=torch.device('cpu'))
decoder = torch.load(modelPath+'decoder_{}_nLayers_{}.pt'.format(gamma, numLayers), map_location=torch.device('cpu'))
decoder.eval()

decoder_dict = decoder.state_dict()
decoder_dict_copy = copy.deepcopy(decoder_dict)
del decoder

decoder = Decoder(10, 512, numLayers, True)
for k in decoder_dict.keys():
    k_old = k
    if len(k.split('module.')) == 2:
        k_new = k.split('module.')[1]
    else:
        k_new = k.split('module.')[0]
    decoder_dict_copy[k_new] = decoder_dict_copy.pop(k_old)

decoder.load_state_dict(decoder_dict_copy)


# Loading Training Data 
trainData = np.load(trainDataPath)
nDim = trainData.shape[1]
trainData = torch.tensor(trainData, dtype=torch.float32)
trainData = trainData[:,None,:]

# Loading Testing Data 
testDataList = []

for param in testParams:
    arr = np.load(testDataPath + 'dataSet_{}.npy'.format(param))
    arr = torch.tensor(arr, dtype=torch.float32)
    arr = arr[:,None,:]
    testDataList.append(arr)
    
    
# Reconstruction on training data
trainLat = encoder(trainData)
trainRecon = decoder(trainLat)
trainLoss = reconLoss(trainRecon.squeeze(), trainData.squeeze()).detach().numpy()

if saveLatent:
    np.save(latentPath + 'dataComb_lat_{}.npy'.format(numLayers), trainLat.detach())
    

# Reconstruction on testing data
testLat = []
testRecon = []
testLoss = []

for i in range(len(testParams)):
    testLat.append(encoder(testDataList[i]))
    testRecon.append(decoder(encoder(testDataList[i])))
    testLoss.append(reconLoss(testRecon[i].squeeze(), testDataList[i].squeeze()).detach().numpy())
    
    if saveLatent:
        np.save(latentPath +  'dataSet_{}_lat_{}.npy'.format(testParams[i], numLayers), testLat[i].detach())

# Comparison plots
fig, ax = plt.subplots(2,len(testParams))
for i in range(2):
    for j in range(len(testParams)):
        
        if i==0:
            CS = ax[i,j].contourf(testDataList[j].squeeze().detach().numpy(), colormap='coolwarm')
            
            if j==0:
                ax[i,j].get_xaxis().set_visible(False)
            else:
                ax[i,j].get_xaxis().set_visible(False)
                ax[i,j].get_yaxis().set_visible(False)
                            
            ax[i,j].set_title('{}'.format(testParams[j]))
            
        else:
            CS = ax[i,j].contourf(testRecon[j].squeeze().detach().numpy(), colormap='coolwarm')

            if j==0:
                continue
            else:
                ax[i,j].get_yaxis().set_visible(False)


            
fig.colorbar(CS, ax=ax.ravel().tolist())
plt.suptitle('$\gamma$={}'.format(gamma))
#ax.axis("off")
plt.savefig(resultsPath + 'Comparison_{}_numLayers_{}.png'.format(gamma,numLayers))
plt.close()


    
# Latent space plots
latDim = int(10)
latArr = testLat[2].detach().squeeze()
#latArr = trainLat[:5000,:].detach().squeeze()

for i in range(latDim):
    plt.plot(latArr[:,i])
    
plt.title('$\gamma$={}'.format(gamma))
plt.xlabel('t')
plt.ylabel('$ \Theta $')
plt.show()
plt.savefig(resultsPath + 'LatPlot_{}_numLayers_{}.png'.format(gamma,numLayers))
plt.close()
 
# Latent space plots
latDim = int(10)
#latArr = testLat[2].detach().squeeze()
latArr = trainLat[:5000,:].detach().squeeze()

for i in range(latDim):
    plt.plot(latArr[:,i])
    
plt.title('$\gamma$={}'.format(gamma))
plt.xlabel('t')
plt.ylabel('$ \Theta $')
plt.show()
plt.savefig(resultsPath + 'LatPlotTrain_{}_numLayers_{}.png'.format(gamma,numLayers))
plt.close()


    
    