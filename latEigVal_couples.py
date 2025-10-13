# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 20:48:50 2023

@author: ashis
"""


# Evaluating latent dimension model - coupled training

import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from torchdiffeq import odeint
import os 
import copy
import torch.distributed as dist


def reconLoss(recon, data):
    
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
    
    def forward(self, t, y):
        # Pass input through hidden layers
        for i in range(self.num_layers):
            y = self.hidden_layers[i](y)
            y = self.activation(y)
        
        # Pass through output layer
        y = self.output_layer(y)
        
        return y


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

def Standardize(data):
    data_mean = torch.mean(data)
    data_std  = torch.std(data)

    std_data = (data - data_mean) / (data_std)

    #print('Mean : {}'.format(data_mean))
    #print('Std  : {}'.format(data_std))

    return data_mean, data_std, std_data

gamma = 0.0
dt = 0.001953125

# File Paths - coupled training 
TrainDataPath    = '../Datasets/Initial_cond/Full Space/dataComb_InitCond.npy'
TestDataPathBase = '../Datasets/Initial_cond/Full Space/'
encoderPathBase  = '../Models/Initial_cond/Coupled nODE/'
decoderPathBase  = '../Models/Initial_cond/Coupled nODE/'
nODEPathBase     = '../Models/Initial_cond/Coupled nODE/'
resultsPathBase  = '../Results/K-S Equations/Initial_cond/Coupled nODE/'
 

# Parameter space (Initial Conditions)
testParams = [23,24,25]
# Hyper-parameters
latDimn = 25
numLayers = 5



# Loading latent space training data 
TrainData = np.load(TrainDataPath)
latDim = TrainData.shape[1]
TrainData = torch.tensor(TrainData, dtype=torch.float32)
#data_mean, data_std, TrainData = Standardize(TrainData)

# Stacking training and target data
nTrainSamp = 6 # Number of training samples
sampSize   = 5000 # Size of each sample

for i in range(nTrainSamp):
    
    if i==0:
        trainData = TrainData[1000:(sampSize), :]
    else:
        trainData = torch.concatenate((trainData, TrainData[(i*sampSize+1000):((i+1)*sampSize), :]))


    
varLengthList = [999]
tLimList_varLen = []
weightNormList = []
EncDecweightNormList = []

# Initialising environment variables
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
dist.init_process_group('gloo')


for varLength in varLengthList:


    print(varLength)

    # Loading Models 
    encoder = torch.load(encoderPathBase+'coupledEncoder_varLen_{}.pt'.format(varLength), map_location=torch.device('cpu'))
    # encoder = torch.load(encoderPathBase+'encoder_0.0_nLayers_5.pt', map_location=torch.device('cpu'))
    encoder.eval()
    encoder_dict = encoder.state_dict()
    encoder_dict_copy = copy.deepcopy(encoder_dict)
    del encoder
    
    encoder = Encoder(latDimn, 512, numLayers, True)
    for k in encoder_dict.keys():
        k_old = k
        if len(k.split('module.')) == 2:
            k_new = k.split('module.')[1]
        else:
            k_new = k.split('module.')[0]
        encoder_dict_copy[k_new] = encoder_dict_copy.pop(k_old)
    
    encoder.load_state_dict(encoder_dict_copy)
    
    # Decoder
    decoder = torch.load(decoderPathBase+'coupledDecoder_varLen_{}.pt'.format(varLength), map_location=torch.device('cpu'))
    # decoder = torch.load(decoderPathBase+'decoder_0.0_nLayers_5.pt', map_location=torch.device('cpu'))
    decoder.eval()
    decoder_dict = decoder.state_dict()
    decoder_dict_copy = copy.deepcopy(decoder_dict)
    del decoder
    
    decoder = Decoder(latDimn, 512, numLayers, True)
    for k in decoder_dict.keys():
        k_old = k
        if len(k.split('module.')) == 2:
            k_new = k.split('module.')[1]
        else:
            k_new = k.split('module.')[0]
        decoder_dict_copy[k_new] = decoder_dict_copy.pop(k_old)

    decoder.load_state_dict(decoder_dict_copy)

    
    # nODE
    
    nODE    = torch.load(nODEPathBase+'couplednODE_varLen_{}.pt'.format(varLength), map_location=torch.device('cpu'))
    # nODE    = torch.load(nODEPathBase+'latModelnODE_0.0_varLen_{}_numLayers_5.pt'.format(varLength), map_location=torch.device('cpu'))
    nODE.eval()

    nODE_dict = nODE.state_dict()
    nODE_dict_copy = copy.deepcopy(nODE_dict)
    del nODE
    
    nODE = FullyConnectedNet(latDimn, latDimn, 120, 5)
    for k in nODE_dict.keys():
        k_old = k
        if len(k.split('module.')) == 2:
            k_new = k.split('module.')[1]
        else:
            k_new = k.split('module.')[0]
        nODE_dict_copy[k_new] = nODE_dict_copy.pop(k_old)

    nODE.load_state_dict(nODE_dict_copy)
    # Eigen value plots 
    # Within training set
    eigsList = []
    tLimList = []
    for i in range(10000,11000):
        currentSamp = trainData[i, :]
        currentSamp = currentSamp[None, None, :]
        
        # Encoding to latent dimension
        currentSampLat = encoder(currentSamp)
        
        # Jacobian
        JacCurr = torch.autograd.functional.jacobian(nODE, (torch.DoubleTensor(0), currentSampLat))
        JacCurr = JacCurr[1]
        JacCurr = JacCurr.squeeze()
        
        # Eigen values
        eigs = torch.linalg.eigvals(JacCurr)
        
        eigs = eigs.numpy()
        
        maxEig = np.amax(eigs.real)
        
        eigsList.append(maxEig)
        
        tLim = 1/maxEig
        
        tLimList.append(tLim)
        
        
    tLimList_varLen.append(tLimList)
        

fullSysTS = []
t = np.linspace(1000,2000,1001)[:1000] * dt
# Loading full-system time scales
with open("tLimList.txt", 'r') as f:
    
    for line in f:
        currLine = line.rstrip('\n')
        fullSysTS.append(float(currLine))
    

# Plotting nODE time-scales 
for i in range(len(varLengthList)):
    plt.semilogy(t, tLimList_varLen[i], label=str(varLengthList[i]+1))
    
# Plotting full-system time-scale
plt.semilogy(t, fullSysTS, label='Full System')
plt.title('Limiting time-scale')
plt.ylabel('$t_{lim}$')
plt.xlabel('t')
plt.legend(title='$n_t$')
plt.savefig(resultsPathBase + 'TimeScaleComp.png')
plt.close()


    

# # Plotting eigen values
# plt.plot(eigsList)
# plt.ylabel('Max eig')
# plt.savefig(resultsPath + 'eigValTraining_{}_varLen_{}.png'.format(gamma, varLength))
# plt.close()

    
# plt.semilogy(tLimList)
# plt.ylabel('Limiting time-scale')
# plt.savefig(resultsPath + 'tLimTraining_{}_varLen_{}.png'.format(gamma, varLength))
# plt.close()


# plt.loglog(varLengthList, weightNormList, label='nODE')
# plt.loglog(varLengthList, EncDecweightNormList, label='Encoder+Decoder')
# plt.xlabel('$n_t$')
# plt.ylabel('$\|$ W $\|_2$')
# plt.legend()
# plt.savefig(resultsPathBase + 'WeightComp.png')
# plt.close()

