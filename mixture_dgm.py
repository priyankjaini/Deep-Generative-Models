'''
This code implements a mixture of deep generative networks
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import math
import matplotlib.pyplot as plt
import seaborn as sns
from generateGMM import *

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# hyperparameters for data generation
n_dims = 1
n_components = 1
n_data = 1000
# hyperparameters for network architecture
input_size = n_dims
hidden_layer1 = 4
hidden_layer2 = 8
hidden_layer3 = 4
output_layer = n_dims
# hyperparameters for network training
num_epochs = 1000
batch_size = 100
learning_rate = 0.01
# define a simple neural network
w11 = torch.randn(n_dims, hidden_layer1, requires_grad=True)
b11 = torch.zeros(1, hidden_layer1, requires_grad=True)
w21 = torch.randn(hidden_layer1, n_dims, requires_grad=True)
b21 = torch.zeros(1, n_dims, requires_grad=True)

w12 = torch.randn(n_dims, hidden_layer1, requires_grad=True)
b12 = torch.zeros(1, hidden_layer1, requires_grad=True)
w22 = torch.randn(hidden_layer1, n_dims, requires_grad=True)
b22 = torch.zeros(1, n_dims, requires_grad=True)

weight1 =torch.randn(1,1, requires_grad=True)
weight2 = 1 - weight1

def forward(x):
  out1 = F.relu(torch.mm(x,w11) + b11)
  out1_final = torch.mm(out1,w21) + b21

  out2 = F.relu(torch.mm(x,w12) + b12)
  out2_final = torch.mm(out2,w22) + b22

  out = weight1*out1_final + weight2*out2_final
  return out


#mean = [0,0]
#cov = [[3,0],[0,1]]
#input_noise = np.random.multivariate_normal(mean, cov, n_data)
mean = 2
cov = 3
input_noise = np.random.normal(mean, cov, n_data)
#print('Input noise is', input_noise.T)
input_noise = torch.from_numpy(input_noise).float()
input_noise = input_noise.reshape(-1, n_dims)
# create neural net model for sample generation
#model_sample_generation = NeuralNet(n_dims)
#original_data = model_sample_generation(input_noise)
#original_data = original_data.detach().numpy()
#data = torch.from_numpy(original_data).float()

criterion = nn.MSELoss()
# TO DO : FIDDLE WITH THE PARAMETERS OF THE OPTIMIZER
optimizer = torch.optim.Adam([w11, b11, w21, b21, w12, b12, w22, b22, weight1], lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

for epoch in range(num_epochs):
    #for i, observations in enumerate(input_noise):
        # move the input_noise to the configured device
        #observations = input_noise.reshape(-1,n_dims)
        # forward pass
        optimizer.zero_grad()
        output = forward(input_noise)
        # loss traces the closeness of the samples generated
        loss = criterion(input_noise,output)
        # backward pass
        loss.backward()
        optimizer.step()

with torch.no_grad():
    #for observations in input_noise_train:
    #observations = observations.reshape(-1,n_dims)
    output = forward(input_noise)
    #print(output)
    #print(input_noise)
    estimated_data = output.detach().numpy()
    input_noise = input_noise.detach().numpy()
    input_min = np.amin(input_noise)
    #print(input_min)
    input_max = np.amax(input_noise)
    #print(input_max)
    bins = np.linspace(input_min, input_max, 10)
    plt.hist(estimated_data, alpha=0.5, label='estimated data')
    plt.hist(input_noise, alpha=0.5, label='original data')
    plt.legend(loc='upper right')
    plt.show()
