# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 19:49:52 2023

@author: ashis
"""

# Model evaluation for coupled training

import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
rc('font', size=22)
rc('legend', fontsize=13)
rc('text.latex', preamble=r'\usepackage{cmbright}')

import numpy as np
from torchdiffeq import odeint
import os 
import torch.distributed as dist
import copy


def RAELoss(recon, data):
    
    return torch.mean(torch.abs(recon-data)/torch.abs(data))

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

# Loss function

def LossConst(recon, data):
        
    # reconLoss = (recon-data)

    # # # Normalizing loss components
    # # reconLoss = (reconLoss - torch.amin(reconLoss)) / (torch.amax(reconLoss) - torch.amin(reconLoss))

    # # MSE Loss
    # gamma = 1.0
    # loss = torch.mean(reconLoss**2) 

    loss = nn.MSELoss()

    # add gamma
    return loss(recon, data)

def Standardize(data):
    data_mean = torch.mean(data)
    data_std  = torch.std(data)

    std_data = (data - data_mean) / (data_std)

    #print('Mean : {}'.format(data_mean))
    #print('Std  : {}'.format(data_std))

    return data_mean, data_std, std_data

def unStandardize(data_mean, data_std, data):
    
    
    unStd_data = (data * data_std) + data_mean
    
    return unStd_data

# Some constants
gamma = 0.0
dt = 0.001953125

# File Paths - coupled training 
# TrainDataPath    = '../Datasets/KSE/dataComb_InitCond.npy'
# # TestDataPathBase = '../Datasets/Initial_cond/Full Space/'
# encoderPathBase  = '../Models/KSE-eq/Coupled/coupledRuns_noL2/'
# decoderPathBase  = '../Models/KSE-eq/Coupled/coupledRuns_noL2/'
# nODEPathBase     = '../Models/KSE-eq/Coupled/coupledRuns_noL2/'
# resultsPathBase  = '../Results/KSE/CoupledFullSpace/'

# File Paths - de-coupled training 
TrainDataPath    = '../Datasets/KSE/dataComb_InitCond.npy'
#TestDataPathBase = '../Datasets/Initial_cond/Full Space/'
encoderPathBase  = '../Models/KSE-eq/DeCoupled/'
decoderPathBase  = '../Models/KSE-eq/DeCoupled/'
nODEPathBase     = '../Models/KSE-eq/DeCoupled/'
resultsPathBase  = '../Results/KSE/CoupledFullSpace/'

# Testing Parameters
testParams = [23,24,25]

# dataset constants
nParams = 22
sampSize = 5000
transPart = 1000
actsampSize = sampSize - transPart


# Loading training data
trainData = np.load(TrainDataPath)
trainDatanoTrans = np.zeros((actsampSize*nParams, trainData.shape[1]))

# Removing the transient part
for i in range(nParams):
    currSamp = trainData[i*sampSize:(i+1)*sampSize,:]
    currSamp = currSamp[transPart:, :]
    trainDatanoTrans[i*actsampSize:(i+1)*actsampSize,:] = currSamp

trainData = trainDatanoTrans
trainData = torch.tensor(trainData, dtype=torch.float32)
trainDataCopy = copy.deepcopy(trainData)

# Standardizing data 
data_mean, data_std, trainData = Standardize(trainData)

# # Loading testing data
# testInputDataList = []
# testInputDataList_copy = []
# testMeanList      = []
# testStdList       = []
# for i in range(len(testParams)):
#     currArr = np.load(TestDataPathBase + 'dataSet_{}.npy'.format(testParams[i]))
#     currArrcopy = copy.deepcopy(torch.tensor(currArr[transPart:, :], dtype=torch.float32))
#     currMean, currStd, currArr = Standardize(torch.tensor(currArr[transPart:, :], dtype=torch.float32))
#     testMeanList.append(currMean)
#     testStdList.append(currStd)
#     testInputDataList.append(currArr)
#     testInputDataList_copy.append(currArrcopy)


varLengthList = [9,49,99,199,499,799,999,1999,3999]
#varLengthList = [99,199,499,799,1999,3999]
#varLengthList = [999]
cutoffSampList = [500, 1000]
#cutoffSampList = [500]
rolloutTrainList = []
rolloutTrainRAEList = []
#rolloutTestList = []
noRolloutTrainList = []
noRolloutTrainRAEList = []
#noRolloutTestList = []
projTrainList = []
#latMagList = []

rolloutTrainList_cutoff = []
rolloutTrainRAEList_cutoff = []
#rolloutTestList_cutoff = []
noRolloutTrainList_cutoff = []
noRolloutTrainRAEList_cutoff = []
#noRolloutTestList_cutoff = []
projTrainList_cutoff = []
#latMagList_cutoff    = []

# Initialising environment variables
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
dist.init_process_group('gloo')

# Hyper-parameters
latDim = 25
numLayers = 5


for cutoffSamp in cutoffSampList:
    print(cutoffSamp)
    
    resultsPath = resultsPathBase + '/cutoff={}/'.format(cutoffSamp)
    # Creating a directory 
    if not os.path.isdir(resultsPath):
        os.mkdir(resultsPath)

    # Resetting lists
    rolloutTrainList = []
    rolloutTrainRAEList = []    
    #rolloutTestList = []
    noRolloutTrainList = []
    noRolloutTrainRAEList = []
    #noRolloutTestList = []
    projTrainList     = []
    #latMagList        = []


    for varLength in varLengthList:
        print(varLength)
        # Loading Models 
        # encoder = torch.load(encoderPathBase+'coupledEncoder_varLen_{}.pt'.format(varLength), map_location=torch.device('cpu'))
        encoder = torch.load(encoderPathBase+'encoder_0.0_nLayers_5.pt', map_location=torch.device('cpu'))
        encoder.eval()
        encoder_dict = encoder.state_dict()
        encoder_dict_copy = copy.deepcopy(encoder_dict)
        del encoder
        
        encoder = Encoder(latDim, 512, numLayers, True)
        for k in encoder_dict.keys():
            k_old = k
            if len(k.split('module.')) == 2:
                k_new = k.split('module.')[1]
            else:
                k_new = k.split('module.')[0]
            encoder_dict_copy[k_new] = encoder_dict_copy.pop(k_old)
        
        encoder.load_state_dict(encoder_dict_copy)
        
        # Decoder
        # decoder = torch.load(decoderPathBase+'coupledDecoder_varLen_{}.pt'.format(varLength), map_location=torch.device('cpu'))
        decoder = torch.load(decoderPathBase+'decoder_0.0_nLayers_5.pt', map_location=torch.device('cpu'))
        decoder.eval()
        decoder_dict = decoder.state_dict()
        decoder_dict_copy = copy.deepcopy(decoder_dict)
        del decoder
        
        decoder = Decoder(latDim, 512, numLayers, True)
        for k in decoder_dict.keys():
            k_old = k
            if len(k.split('module.')) == 2:
                k_new = k.split('module.')[1]
            else:
                k_new = k.split('module.')[0]
            decoder_dict_copy[k_new] = decoder_dict_copy.pop(k_old)

        decoder.load_state_dict(decoder_dict_copy)

        
        # nODE
        
        # nODE    = torch.load(nODEPathBase+'couplednODE_varLen_{}.pt'.format(varLength), map_location=torch.device('cpu'))
        nODE    = torch.load(nODEPathBase+'latModelnODE_0.0_varLen_{}_numLayers_5.pt'.format(varLength), map_location=torch.device('cpu'))
        nODE.eval()

        nODE_dict = nODE.state_dict()
        nODE_dict_copy = copy.deepcopy(nODE_dict)
        del nODE
        
        nODE = FullyConnectedNet(latDim, latDim, 120, 5)
        for k in nODE_dict.keys():
            k_old = k
            if len(k.split('module.')) == 2:
                k_new = k.split('module.')[1]
            else:
                k_new = k.split('module.')[0]
            nODE_dict_copy[k_new] = nODE_dict_copy.pop(k_old)

        nODE.load_state_dict(nODE_dict_copy)
        
        ### Just reconstruction
        # Encode 
        trainData = copy.deepcopy(trainDataCopy)
        data_mean, data_std, trainData = Standardize(trainData)
        trainData = trainData[:, None, :]
        Lat = encoder(trainData)
        trainData = trainData.squeeze()
        
        # Decode
        reconProj = decoder(Lat)
        reconProj = reconProj.squeeze()
        reconProj = unStandardize(data_mean, data_std, reconProj)
        
        # Projection error
        MSE_loss = torch.nn.MSELoss()
        lossProj = MSE_loss(reconProj[:cutoffSamp,:], trainDataCopy[:cutoffSamp,:])
        projTrainList.append(copy.deepcopy(lossProj.detach()*1e-5))
        
        # del trainData, reconProj, data_mean, data_std
        
        # ### Rollout Testing 
        # testEvolveList = []
        # dtTest = dt
        # t_batch = dt * (torch.linspace(0,actsampSize,actsampSize+1)[:(actsampSize)])
        # t_batch_Rollout = t_batch
        
        # ## Within training set 
        
        # # Encoding to latent space
        
        # # Standardizing
        # trainData = copy.deepcopy(trainDataCopy)
        # data_mean, data_std, trainData = Standardize(trainData)
        # trainData = trainData[:, None, :]
        # y0 = encoder(trainData)
        # trainData = trainData.squeeze()
        
        # # Extracting only intital conditions
        # y0 = y0[0::actsampSize, :]
        
        # # nODE rollout
        # LatreconArray = odeint(nODE, y0, t_batch, method='euler')
        # LatreconArray = LatreconArray.flatten(start_dim=0, end_dim=1)
        # #latMagList.append(torch.amax(torch.amax(LatreconArray[:cutoffSamp,:].detach())))
        
        # # Decoding to full space
        # reconArray = decoder(LatreconArray[:cutoffSamp,:])
        # reconArray = reconArray.squeeze()
        
        # # De-standardizing data 
        # reconArray = unStandardize(data_mean, data_std, reconArray)
        
        
        # # MSE- Loss
        # MSE_loss = torch.nn.MSELoss()
        # # Reconstruction 'roll-out' loss
        # lossRecon = MSE_loss(reconArray[:cutoffSamp,:], trainDataCopy[:cutoffSamp,:])
        # # Last sample
        # #lossRecon = MSE_loss(reconArray[cutoffSamp-1,:], trainData[cutoffSamp-1,:])
        # rolloutTrainList.append(lossRecon.detach())
        # rolloutTrainRAEList.append(RAELoss(reconArray[:cutoffSamp,:], trainDataCopy[:cutoffSamp,:]).detach())
        
        # # Comparison Plots - full space
        # # Comparison plots
        # fig, ax = plt.subplots(2,1)
        # for i in range(2):
        #     for j in range(len(testParams)):
                
        #         if i==0:
        #             CS = ax[i].contourf(trainDataCopy[:cutoffSamp, :])
                    
        #             if j==0:
        #                 ax[i].get_xaxis().set_visible(False)
        #             else:
        #                 ax[i].get_xaxis().set_visible(False)
        #                 ax[i].get_yaxis().set_visible(False)
                                    
        #             ax[i].set_title('{}'.format(testParams[j]))
                    
        #         else:
        #             CS = ax[i].contourf(reconArray[:cutoffSamp, :].detach())
        
        #             if j==0:
        #                 continue
        #             else:
        #                 ax[i].get_yaxis().set_visible(False)

        # fig.colorbar(CS, ax=ax.ravel().tolist())
        # plt.suptitle('Rollout: Training Set')
        # plt.savefig(resultsPath + 'RolloutTrain_{}_varLen_{}.png'.format(gamma, varLength))
        # plt.close()
        
        # # Latent Space Trajectories
        # for k in range(10):
        #     plt.plot(t_batch_Rollout[:cutoffSamp], LatreconArray[:cutoffSamp,k].detach())

        # plt.title('Latent Space Trajectories')
        # plt.xlabel('t')
        # plt.ylabel('$\theta$')
        # plt.savefig(resultsPath + 'RollOutTrain_{}_LatTraj_varLen_{}.png'.format(gamma, varLength))
        # plt.close()
        
        # # Deleting arrays
        # del LatreconArray, reconArray, trainData, data_mean, data_std
        
        ## Within testing set
        # testLoss = 0
        # for i in range(len(testParams)):
        
        #     # Encoding to latent space
        #     testData = copy.deepcopy(testInputDataList[i])
        #     testDataCopy = copy.deepcopy(testInputDataList_copy[i])
        #     testData = testData[:, None, :]
        #     y0 = encoder(testData)
        #     testData = testData.squeeze()
            
        #     # Extracting only intital conditions
        #     y0 = y0[0::actsampSize, :]
            
        #     # nODE rollout
        #     LatreconArray = odeint(nODE, y0, t_batch, method='euler')
        #     LatreconArray = LatreconArray.flatten(start_dim=0, end_dim=1)
            
        #     # Decoding to full space
        #     reconArray = decoder(LatreconArray)
        #     reconArray = reconArray.squeeze()
            
        #     # Un-standardizing data
        #     reconArray = unStandardize(testMeanList[i], testStdList[i], reconArray)
            
        #     # MSE- Loss
        #     MSE_loss = torch.nn.MSELoss()
            
        #     # Reconstruction 'roll-out' loss
        #     lossRecon = MSE_loss(reconArray[:cutoffSamp,:], testDataCopy[:cutoffSamp,:])
        #     # Last sample
        #     #lossRecon = MSE_loss(reconArray[cutoffSamp-1,:], trainData[cutoffSamp-1,:])
        #     testLoss += (1/len(testParams)) * lossRecon
            
        # rolloutTestList.append(testLoss.detach())
        
        # # Comparison Plots
        # # Comparison Plots - full space
        # # Comparison plots
        # fig, ax = plt.subplots(2,len(testParams))
        # for i in range(2):
        #     for j in range(len(testParams)):
                
        #         if i==0:
        #             CS = ax[i,j].contourf(testDataCopy[:cutoffSamp, :])
                    
        #             if j==0:
        #                 ax[i,j].get_xaxis().set_visible(False)
        #             else:
        #                 ax[i,j].get_xaxis().set_visible(False)
        #                 ax[i,j].get_yaxis().set_visible(False)
                                    
        #             ax[i,j].set_title('{}'.format(testParams[j]))
                    
        #         else:
        #             CS = ax[i,j].contourf(reconArray[:cutoffSamp, :].detach())
        
        #             if j==0:
        #                 continue
        #             else:
        #                 ax[i,j].get_yaxis().set_visible(False)

        # fig.colorbar(CS, ax=ax.ravel().tolist())
        # plt.suptitle('Rollout: Testing Set')
        # plt.savefig(resultsPath + 'RolloutTest_{}_varLen_{}.png'.format(gamma, varLength))
        # plt.close()
        
        # del reconArray, testData, testDataCopy
                
            
        # ## No-Rollout testing
        
        # ## Within training set
        # # Setting t_batch
        # actsampSize = sampSize - transPart
        # testEvolveList = []
        # dtTest = dt
        # t_batch = dt * (torch.linspace(0,1,2))

        
        # # Encoding to latent space
        # trainData = copy.deepcopy(trainDataCopy)
        # data_mean, data_std, trainData = Standardize(trainData)
        # trainData = trainData[:, None, :]
        # y0 = encoder(trainData)
        # trainData = trainData.squeeze()

        # # nODE rollout
        # LatreconArray = odeint(nODE, y0, t_batch, method='euler')
        # LatreconArray = LatreconArray[1,:,:]
        
        # # Decoding to full space
        # reconArray = decoder(LatreconArray)
        # reconArray = reconArray.squeeze()
        
        # # Un-standardizing data 
        # reconArray = unStandardize(data_mean, data_std, reconArray)
        
        # # MSE- Loss
        # MSE_loss = torch.nn.MSELoss()
        # # Reconstruction 'roll-out' loss
        # lossRecon = MSE_loss(reconArray[:cutoffSamp-1,:], trainDataCopy[1:cutoffSamp,:])
        # noRolloutTrainList.append(lossRecon.detach())
        # noRolloutTrainRAEList.append(RAELoss(reconArray[:cutoffSamp-1,:], trainDataCopy[1:cutoffSamp,:]).detach())
        
        # # Comparison Plots - full space
        # # Comparison plots
        # fig, ax = plt.subplots(2,1)
        # for i in range(2):
        #     for j in range(len(testParams)):
                
        #         if i==0:
        #             CS = ax[i].contourf(trainDataCopy[1:cutoffSamp, :])
        #             np.save(resultsPath + 'trainData_{}_varLen_{}.npy'.format(gamma, varLength),trainDataCopy[1:cutoffSamp, :])
                    
        #             if j==0:
        #                 ax[i].get_xaxis().set_visible(False)
        #             else:
        #                 ax[i].get_xaxis().set_visible(False)
        #                 ax[i].get_yaxis().set_visible(False)
                                    
        #             #ax[i].set_title('{}'.format(testParams[j]))
                    
        #         else:
        #             CS = ax[i].contourf(reconArray[:cutoffSamp-1, :].detach())
        #             np.save(resultsPath + 'reconArray_{}_varLen_{}.npy'.format(gamma, varLength),reconArray[:cutoffSamp-1, :].detach())

        #             if j==0:
        #                 continue
        #             else:
        #                 ax[i].get_yaxis().set_visible(False)

        # fig.colorbar(CS, ax=ax.ravel().tolist())
        # ax[0].set_title('Ground Truth')
        # ax[1].set_title('Reconstruction')
        # ax[i].get_yaxis().set_visible(True)
        # ax[1].set_ylabel('t')
        # ax[1].set_xlabel('x')
        # #plt.suptitle('No Rollout: Training Set')
        # plt.savefig(resultsPath + 'NoRolloutTrain_{}_varLen_{}.pdf'.format(gamma, varLength))
        # plt.close()
        
        # # Latent Space Trajectories
        # for k in range(10):
        #     plt.plot(t_batch_Rollout[1:cutoffSamp], LatreconArray[:cutoffSamp-1, k].detach())

        # plt.title('Latent Space Trajectories')
        # plt.xlabel('t')
        # plt.ylabel('$\theta$')
        # plt.savefig(resultsPath + 'NoRollOutTrain_{}_LatTraj_varLen_{}.png'.format(gamma, varLength))
        # plt.close()

        # del reconArray, LatreconArray, trainData, data_mean, data_std      

        # ## Within testing set
        # testLoss = 0
        # for i in range(len(testParams)):
        
        #     # Encoding to latent space
        #     testData = copy.deepcopy(testInputDataList[i])
        #     testDataCopy = copy.deepcopy(testInputDataList_copy[i])
        #     testData = testData[:, None, :]
        #     y0 = encoder(testData)
        #     testData = testData.squeeze()
            
            
        #     # nODE rollout
        #     LatreconArray = odeint(nODE, y0, t_batch, method='euler')
        #     LatreconArray = LatreconArray[1,:,:]
            
        #     # Decoding to full space
        #     reconArray = decoder(LatreconArray)
        #     reconArray = reconArray.squeeze()
            
        #     # Un-standadizing data 
        #     reconArray = unStandardize(testMeanList[i], testStdList[i], reconArray)
            
        #     # MSE- Loss
        #     MSE_loss = torch.nn.MSELoss()
            
        #     # Reconstruction 'roll-out' loss
        #     lossRecon = MSE_loss(reconArray[:cutoffSamp-1,:], testDataCopy[1:cutoffSamp,:])
        
        #     testLoss += (1/len(testParams)) * lossRecon

        # noRolloutTestList.append(testLoss.detach())
        
        # # Comparison Plots
        # # Comparison Plots - full space
        # # Comparison plots
        # fig, ax = plt.subplots(2,len(testParams))
        # for i in range(2):
        #     for j in range(len(testParams)):
                
        #         if i==0:
        #             CS = ax[i,j].contourf(testDataCopy[1:cutoffSamp, :])
                    
        #             if j==0:
        #                 ax[i,j].get_xaxis().set_visible(False)
        #             else:
        #                 ax[i,j].get_xaxis().set_visible(False)
        #                 ax[i,j].get_yaxis().set_visible(False)
                                    
        #             ax[i,j].set_title('{}'.format(testParams[j]))
                    
        #         else:
        #             CS = ax[i,j].contourf(reconArray[:cutoffSamp-1, :].detach())
        
        #             if j==0:
        #                 continue
        #             else:
        #                 ax[i,j].get_yaxis().set_visible(False)

        # fig.colorbar(CS, ax=ax.ravel().tolist())
        # plt.suptitle('No Rollout: Testing Set')
        # plt.savefig(resultsPath + 'NoRolloutTest_{}_varLen_{}.png'.format(gamma, varLength))
        # plt.close()
        
        # del reconArray
        
    # rolloutTrainList_cutoff.append(rolloutTrainList)
    # rolloutTrainRAEList_cutoff.append(rolloutTrainRAEList)
    #rolloutTestList_cutoff.append(rolloutTestList)
    # noRolloutTrainList_cutoff.append(noRolloutTrainList)
    # noRolloutTrainRAEList_cutoff.append(noRolloutTrainRAEList)
    #noRolloutTestList_cutoff.append(noRolloutTestList)
    projTrainList_cutoff.append(projTrainList)
    #latMagList_cutoff.append(latMagList)
        
# # Adding physical time to varLength
# for i in range(len(varLengthList)):
#     varLengthList[i] *= dt

# # Plotting combined plots
# # Rollout Training Loss
# fig, ax1 = plt.subplots()
# color = 'tab:brown'
# for i in range(len(cutoffSampList)):
#     ax1.loglog(varLengthList, rolloutTrainList_cutoff[i],  color=color)

# ax1.set_xlabel('$n_t$')
# ax1.set_ylabel('MSE', color=color)


# ax2 = ax1.twinx()
# color = 'tab:pink'
# for i in range(len(cutoffSampList)):
#     ax2.loglog(varLengthList, rolloutTrainRAEList_cutoff[i], color=color)

# ax2.set_ylabel('RAE', color=color)
# #plt.title('Rollout: Training set')
# #plt.legend(title='Rollout Length')
# fig.tight_layout()
# plt.savefig(resultsPathBase+'RollTrainLossvsLen.png')
# plt.close()

# # Rollout Testing Loss
# for i in range(len(cutoffSampList)):
#     plt.loglog(varLengthList, rolloutTestList_cutoff[i], label=str(cutoffSampList[i]))

# plt.xlabel('$n_t$')
# plt.ylabel('Loss')
# plt.title('Rollout: Testing set')
# #plt.ylim([5e1,1e4])
# plt.legend(title='Rollout Length')
# plt.savefig(resultsPathBase+'RollTestLossvsLen.png')
# plt.close()


#Projection Loss
for i in range(len(cutoffSampList)):
    plt.loglog(varLengthList, projTrainList_cutoff[i], label=str(cutoffSampList[i]))

plt.xlabel('$n_t$')
plt.ylabel('$\mathcal{L}_{AE}$')
plt.legend(title='$n_{RO}$')
plt.tight_layout()
plt.savefig(resultsPathBase+'ProjTrainLossvsLen_deCoupled.pdf')
plt.close()

# # No-Rollout Training Loss
# fig, ax1 = plt.subplots()
# color = 'tab:brown'
# for i in range(len(cutoffSampList)):
#     ax1.loglog(varLengthList, noRolloutTrainList_cutoff[i],  color=color)

# ax1.set_xlabel('$n_t$')
# ax1.set_ylabel('MSE', color=color)


# ax2 = ax1.twinx()
# color = 'tab:pink'
# for i in range(len(cutoffSampList)):
#     ax2.loglog(varLengthList, noRolloutTrainRAEList_cutoff[i], color=color)

# ax2.set_ylabel('RAE', color=color)
# #plt.title('Rollout: Training set')
# #plt.legend(title='Rollout Length')
# fig.tight_layout()
# plt.savefig(resultsPathBase+'NoRollTrainLossvsLen.png')
# plt.close()


# # No-Rollout Testting Loss
# for i in range(len(cutoffSampList)):
#     plt.loglog(varLengthList, noRolloutTestList_cutoff[i], label=str(cutoffSampList[i]))

# plt.xlabel('$n_t$')
# plt.ylabel('Loss')
# plt.title('No-Rollout: Testing set')
# plt.legend(title='Rollout Length')
# plt.savefig(resultsPathBase+'NoRollTestLossvsLen.png')
# plt.close()
        
# # Magnitude of latent space
# for i in range(len(cutoffSampList)):
#     plt.loglog(varLengthList, latMagList_cutoff[i], label=str(cutoffSampList[i]))

# plt.xlabel('$n_t$')
# plt.ylabel('Loss')
# plt.title('Latent Space Magnitude')
# plt.legend(title='Rollout Length')
# plt.savefig(resultsPathBase+'LatSpaceMagvsLen.png')
# plt.close()     