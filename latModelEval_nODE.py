# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 22:31:54 2023

@author: ashis
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 15:39:37 2023

@author: ashis
"""

# Evaluating latent dimension model

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



def reconLoss(recon, data):
    
    return (torch.norm(recon-data)) / latDim



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


gamma = 0.0
dt = 0.001953125
numLayers = 5

# File Paths
# modelPath = '../Models/KSE-eq/Coupled/coupledRuns/'
modelPath = '../Models/KSE-eq/DeCoupled/'
LatTrainDataPath = '../Datasets/KSE/dataComb_lat_{}.npy'.format(numLayers)
# LatTestDataPath  = '../Datasets/Initial_cond/Latent Space/numLayers_Test/'
# resultsPathBase   = '../Results/KSE/Coupled/'
resultsPathBase   = '../Results/KSE/DeCoupledRecheck/'
 

# Parameter space (Initial Conditions)
testParams = [23,24,25]



# Loading latent space training data 
LatTrainData = np.load(LatTrainDataPath)
latDim = LatTrainData.shape[1]
LatTrainData = torch.tensor(LatTrainData, dtype=torch.float32)

# Stacking training and target data
nTrainSamp = 22 # Number of training samples
sampSize   = 5000 # Size of each sample
transPart = 1000 # Cut-off for transient part



trainData = torch.zeros( sampSize-transPart, nTrainSamp, latDim)
for i in range(nTrainSamp):
    
    # if i==0:
    #     trainData[i,:,:] = LatTrainData[transPart:(sampSize), :]
    # else:
    #     trainData[] = torch.concatenate((trainData, LatTrainData[(i*sampSize+transPart):((i+1)*sampSize), :]))

    trainData[:,i,:] = LatTrainData[(i*sampSize+transPart):((i+1)*sampSize), :]


# # Stacking target data 
# testInputDataList = []
# for i in range(len(testParams)):
    
#     latArray = np.load(LatTestDataPath + 'dataSet_{}_lat_{}.npy'.format(testParams[i], numLayers))
#     testInputDataList.append(torch.tensor(latArray[1000:sampSize,:], dtype=torch.float32))
    

varLengthList = [9,49,99,199,499,799,999,1999,3999]
cutoffSampList = [250, 500, 750, 1000, 1500, 2000, 3000, 4000]
rolloutTrainList = []
rolloutTrainRAEList = []
# rolloutTestList = []
noRolloutTrainList = []
noRollouttrainRAEList = []
# noRolloutTestList = []

rolloutTrainList_cutoff = []
rolloutTrainRAEList_cutoff = []
#rolloutTestList_cutoff = []
noRolloutTrainList_cutoff = []
noRolloutTrainRAEList_cutoff = []
#noRolloutTestList_cutoff = []

# Initialising environment variables
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
dist.init_process_group('gloo')



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
    
    
    
    for varLength in varLengthList:



        ### Loading Model
        print(varLength)
        # varLength = 499
        # cutoffSamp = 1000
        # nODE
        
        #nODE    = torch.load(modelPath+'couplednODE_varLen_{}.pt'.format(varLength), map_location=torch.device('cpu'))
        nODE    = torch.load(modelPath+'latModelnODE_0.0_varLen_{}_numLayers_5.pt'.format(varLength), map_location=torch.device('cpu'))
        nODE.eval()

        nODE_dict = nODE.state_dict()
        nODE_dict_copy = copy.deepcopy(nODE_dict)
        del nODE
        
        model = FullyConnectedNet(25, 25, 120, 5)
        for k in nODE_dict.keys():
            k_old = k
            if len(k.split('module.')) == 2:
                k_new = k.split('module.')[1]
            else:
                k_new = k.split('module.')[0]
            nODE_dict_copy[k_new] = nODE_dict_copy.pop(k_old)

        model.load_state_dict(nODE_dict_copy)
         
        
        ### Rollout Testing  
        # Evolving latent trajectories
        actsampSize = sampSize - transPart
        testEvolveList = []
        dtTest = dt
        t_batch = dt * (torch.linspace(0,actsampSize,actsampSize+1)[:(actsampSize)])
        t_batch_Rollout = t_batch
        
        
        # Reconstruction
        y0 = trainData[0,:,:]
        #y0       = y0[None,:]
        reconArray = odeint(model, y0, t_batch, method='euler')
        reconArray = reconArray.squeeze()
        
        
        # MSE- Loss
        MSE_loss = torch.nn.MSELoss()
        # Reconstruction 'roll-out' loss
        lossRecon = MSE_loss(reconArray[:cutoffSamp,:,:], trainData[:cutoffSamp,:,:])
        rolloutTrainList.append(lossRecon.detach())
        rolloutTrainRAEList.append(RAELoss(reconArray[:cutoffSamp,:,:], trainData[:cutoffSamp,:,:]).detach())
        #print('Recon-rollout Loss : {}'.format(lossRecon))
        
        # Comparison plots
        fig, ax = plt.subplots(2,1)
        for i in range(2):
            for j in range(len(testParams)):
                
                if i==0:
                    for k in range(latDim):
                        ax[i].plot(t_batch[:cutoffSamp], trainData[:cutoffSamp,0,k])
                    np.save(resultsPath + 'latTrain_{}_varLen_{}_roll.npy'.format(gamma, varLength), trainData[1:cutoffSamp,0,:])
                    if j==0:
                        ax[i].get_xaxis().set_visible(False)
                    else:
                        ax[i].get_xaxis().set_visible(False)
                        ax[i].get_yaxis().set_visible(False)
                                    
                    # ax[i].set_title('{}'.format(testParams[j]))
                    
                else:
                    for k in range(latDim):
                        ax[i].plot(t_batch[:cutoffSamp],reconArray[:cutoffSamp,0,k].detach())
                    np.save(resultsPath + 'latRecon_{}_varLen_{}_roll.npy'.format(gamma, varLength), reconArray[1:cutoffSamp,0,:].detach())
                    if j==0:
                        continue
                    else:
                        ax[i].get_yaxis().set_visible(False)
        
        plt.suptitle('Rollout: Training Set')
        #ax.axis("off")
        plt.savefig(resultsPath + 'RolloutTrain_{}_varLen_{}.png'.format(gamma, varLength))
        plt.close()
        
        del reconArray, y0
        
        # # Test
        # for i in range(len(testParams)):
            
        #     latArray = testInputDataList[i]
        #     y0       = latArray[0,:]
        #     y0       = y0[None,:]
        #     y        = y0
            
        #     reconArray = odeint(model, y0, t_batch, method='euler')
                
        #     testEvolveList.append(reconArray.squeeze())
        #     del reconArray, y0
        
        
        # # Comparison plots
        # testLoss = 0
        # fig, ax = plt.subplots(2,len(testParams))
        # for i in range(2):
        #     for j in range(len(testParams)):
                
        #         if i==0:
        #             for k in range(latDim):
        #                 ax[i,j].plot(t_batch[:cutoffSamp],testInputDataList[j][:cutoffSamp,k])
                    
        #             if j==0:
        #                 ax[i,j].get_xaxis().set_visible(False)
        #             else:
        #                 ax[i,j].get_xaxis().set_visible(False)
        #                 ax[i,j].get_yaxis().set_visible(False)
                                    
        #             ax[i,j].set_title('{}'.format(testParams[j]))
                    
        #         else:
        #             for k in range(latDim):
        #                 ax[i,j].plot(t_batch[:cutoffSamp],testEvolveList[j][:cutoffSamp,k].detach())
        
        #             if j==0:
        #                 continue
        #             else:
        #                 ax[i,j].get_yaxis().set_visible(False)
                        
        #         testLoss = testLoss + (1/len(testParams)) * MSE_loss(testInputDataList[j][:cutoffSamp,:], testEvolveList[j][:cutoffSamp,:])
        
        # #print('Test-rollout Loss : {}'.format(testLoss))           
        # rolloutTestList.append(testLoss.detach())
        # plt.suptitle('Rollout: Testing Set')
        # #ax.axis("off")
        # plt.savefig(resultsPath + 'RolloutTest_{}_varLen_{}.png'.format(gamma, varLength))
        # plt.close()
        
        
        ### Non-rollout testing
        # Evolving latent trajectories
        actsampSize = sampSize - transPart
        testEvolveList = []
        dtTest = dt
        t_batch = dt * (torch.linspace(0,1,2))
        
        
        
        # Reconstruction
        y0 = trainData[:, :actsampSize, :].flatten(start_dim=0, end_dim=1)
        reconArray = odeint(model, y0, t_batch, method='euler')
        reconArray = reconArray[1,:,:].unflatten(0, (actsampSize, nTrainSamp))
        
        
        # MSE- Loss
        MSE_loss = torch.nn.MSELoss()
        # Reconstruction 'roll-out' loss
        lossRecon = MSE_loss(reconArray[:cutoffSamp-1,:,:], trainData[1:cutoffSamp,:,:])
        noRolloutTrainList.append(lossRecon.detach())
        noRolloutTrainRAEList.append(RAELoss(reconArray[:cutoffSamp-1,:,:], trainData[1:cutoffSamp,:,:]).detach())
        #print('Recon-rollout Loss : {}'.format(lossRecon))
        
        # Comparison plots
        fig, ax = plt.subplots(2,1)
        for i in range(2):
            for j in range(len(testParams)):
                
                if i==0:
                    for k in range(latDim):
                        ax[i].plot(trainData[1:cutoffSamp,0,k])
                    np.save(resultsPath + 'latTrain_{}_varLen_{}.npy'.format(gamma, varLength), trainData[1:cutoffSamp,0,:])
                    
                    if j==0:
                        ax[i].get_xaxis().set_visible(False)
                    else:
                        ax[i].get_xaxis().set_visible(False)
                        ax[i].get_yaxis().set_visible(False)
                                    
                    # ax[i].set_title('{}'.format(testParams[j]))
                    
                else:
                    for k in range(latDim):
                        ax[i].plot(reconArray[:cutoffSamp-1,0,k].detach())
                    np.save(resultsPath + 'latRecon_{}_varLen_{}.npy'.format(gamma, varLength), reconArray[1:cutoffSamp,0,:].detach())
        
                    if j==0:
                        continue
                    else:
                        ax[i].get_yaxis().set_visible(False)
        
        #plt.suptitle('No-Rollout: Training Set')
        #ax.axis("off")
        ax[0].set_title('Ground Truth')
        ax[1].set_title('Reconstruction')
        ax[i].get_yaxis().set_visible(True)
        ax[1].set_ylabel('$\Theta$')
        ax[1].set_xlabel('x')

        plt.savefig(resultsPath + 'NoRolloutTrain_{}_varLen_{}.pdf'.format(gamma, varLength))
        plt.close()
        
        del reconArray, y0
        
        # # Test
        # for i in range(len(testParams)):
            
        #     latArray = testInputDataList[i]
        #     y0       = latArray[:actsampSize,:]
            
        #     reconArray = odeint(model, y0, t_batch, method='euler')
        #     reconArray = reconArray[1,:,:]
                
        #     testEvolveList.append(reconArray.squeeze())
        #     del reconArray, y0
        
        
        # # Comparison plots
        # testLoss = 0
        # fig, ax = plt.subplots(2,len(testParams))
        # for i in range(2):
        #     for j in range(len(testParams)):
                
        #         if i==0:
        #             for k in range(latDim):
        #                 ax[i,j].plot(t_batch_Rollout[1:cutoffSamp],testInputDataList[j][1:cutoffSamp,k])
                    
        #             if j==0:
        #                 ax[i,j].get_xaxis().set_visible(False)
        #             else:
        #                 ax[i,j].get_xaxis().set_visible(False)
        #                 ax[i,j].get_yaxis().set_visible(False)
                                    
        #             ax[i,j].set_title('{}'.format(testParams[j]))
                    
        #         else:
        #             for k in range(latDim):
        #                 ax[i,j].plot(t_batch_Rollout[1:cutoffSamp],testEvolveList[j][:cutoffSamp-1,k].detach())
        
        #             if j==0:
        #                 continue
        #             else:
        #                 ax[i,j].get_yaxis().set_visible(False)
                        
        #         testLoss = testLoss + (1/len(testParams)) * MSE_loss(testInputDataList[j][1:cutoffSamp+1,:], testEvolveList[j][:cutoffSamp,:])
        
        # #print('Test-rollout Loss : {}'.format(testLoss))           
        # noRolloutTestList.append(testLoss.detach())
        # plt.suptitle('No-Rollout: Testing Set')
        # #ax.axis("off")
        # plt.savefig(resultsPath + 'NoRolloutTest_{}_varLen_{}.png'.format(gamma, varLength))
        # plt.close()

    rolloutTrainList_cutoff.append(rolloutTrainList)
    rolloutTrainRAEList_cutoff.append(rolloutTrainRAEList)
    # rolloutTestList_cutoff.append(rolloutTestList)
    noRolloutTrainList_cutoff.append(noRolloutTrainList)
    noRolloutTrainRAEList_cutoff.append(noRolloutTrainRAEList)
    # noRolloutTestList_cutoff.append(noRolloutTestList)



# # Adding physical time to varLength
# for i in range(len(varLengthList)):
#     varLengthList[i] *= dt

# Plotting combined plots
# Rollout Training Loss
fig, ax1 = plt.subplots()
color = 'black'
for i in range(len(cutoffSampList)):
    ax1.loglog(varLengthList, rolloutTrainList_cutoff[i],  label=str(cutoffSampList[i]))

ax1.legend(title='$n_{RO}$')
ax1.set_xlabel('$n_t$')
ax1.set_ylabel('$\mathcal{L}_{{RO}}$', color=color)


# ax2 = ax1.twinx()
# color = 'red'
# for i in range(len(cutoffSampList)):
#     ax2.loglog(varLengthList, rolloutTrainRAEList_cutoff[i], color=color)

# ax2.set_ylabel('$\mathcal{R}_{{RO}}$', color=color)
# #plt.title('Rollout: Training set')
# #plt.legend(title='Rollout Length')
fig.tight_layout()
plt.savefig(resultsPathBase+'RollTrainLossvsLen.pdf')
plt.close()

# # Rollout Testing Loss
# for i in range(len(cutoffSampList)):
#     plt.loglog(varLengthList, rolloutTestList_cutoff[i], label=str(cutoffSampList[i]))

# plt.xlabel('$n_t$')
# plt.ylabel('Loss')
# plt.title('Rollout: Testing set')
# plt.legend(title='Rollout Length')
# plt.savefig(resultsPathBase+'RollTestLossvsLen.png')
# plt.close()

# No-Rollout Training Loss
fig, ax1 = plt.subplots()
# color = 'black'
for i in range(len(cutoffSampList)):
    ax1.loglog(varLengthList, noRolloutTrainList_cutoff[i],  label=str(cutoffSampList[i]))

ax1.legend(title='$n_{RO}$')
ax1.set_xlabel('$n_t$')
ax1.set_ylabel('$\mathcal{L}_{{SS}}$', color=color)


# ax2 = ax1.twinx()
# color = 'red'
# for i in range(len(cutoffSampList)):
#     ax2.loglog(varLengthList, noRolloutTrainRAEList_cutoff[i], color=color)

# ax2.set_ylabel('$\mathcal{R}_{{SS}}$', color=color)
#plt.title('Rollout: Training set')
#plt.legend(title='Rollout Length')
fig.tight_layout()
plt.savefig(resultsPathBase+'NoRollTrainLossvsLen.pdf')
plt.close()



# # No-Rollout Testting Loss
# for i in range(len(cutoffSampList)):
#     plt.loglog(varLengthList, noRolloutTestList_cutoff[i], label=str(cutoffSampList[i]))

# plt.xlabel('$n_t$')
# plt.ylabel('Loss')
# plt.title('No-Rollout: Testing set')
# plt.legend(title='Rollout Length')
# plt.savefig(resultsPathBase+'NoRollTestLossvsLen.png')
# plt.close()

# # Eigen value plots 
# # Within training set
# eigsList = []
# tLimList = []
# for i in range(1000,2000):
#     currentSamp = trainData[i, :]
#     currentSamp = currentSamp[None, :]
    
#     # Jacobian
#     JacCurr = torch.autograd.functional.jacobian(model, (torch.DoubleTensor(0), currentSamp))
#     JacCurr = JacCurr[1]
#     JacCurr = JacCurr.squeeze()
    
#     # Eigen values
#     eigs = torch.linalg.eigvals(JacCurr)
    
#     eigs = eigs.numpy()
    
#     maxEig = np.amax(eigs.real)
    
#     eigsList.append(maxEig)
    
#     tLim = 1/maxEig
    
#     tLimList.append(tLim)
    
    
# # Plotting eigen values
# plt.plot(eigsList)
# plt.ylabel('Max eig')
# plt.savefig(resultsPath + 'eigValTraining_{}_varLen_{}.png'.format(gamma, varLength))
# plt.close()

    
# plt.semilogy(tLimList)
# plt.ylabel('Limiting time-scale')
# plt.savefig(resultsPath + 'tLimTraining_{}_varLen_{}.png'.format(gamma, varLength))
# plt.close()

    
    
    
    
    
    
    
    
    






