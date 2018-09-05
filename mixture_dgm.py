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
hidden_layer1 = 1
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
'''
w12 = torch.randn(n_dims, hidden_layer1, requires_grad=True)
b12 = torch.zeros(1, hidden_layer1, requires_grad=True)
w22 = torch.randn(hidden_layer1, n_dims, requires_grad=True)
b22 = torch.zeros(1, n_dims, requires_grad=True)

w13 = torch.randn(n_dims, hidden_layer1, requires_grad=True)
b13 = torch.zeros(1, hidden_layer1, requires_grad=True)
w23 = torch.randn(hidden_layer1, n_dims, requires_grad=True)
b23 = torch.zeros(1, n_dims, requires_grad=True)

w14 = torch.randn(n_dims, hidden_layer1, requires_grad=True)
b14 = torch.zeros(1, hidden_layer1, requires_grad=True)
w24 = torch.randn(hidden_layer1, n_dims, requires_grad=True)
b24 = torch.zeros(1, n_dims, requires_grad=True)

weight1 =torch.randn(1,1, requires_grad=True)
weight2 =torch.randn(1,1, requires_grad=True)
weight3 =torch.randn(1,1, requires_grad=True)
weight4 = 1 - (weight1 + weight2 + weight3)
'''
def forward(x):
  out1 = F.tanh(torch.mm(x,w11) + b11)
  #out1 = torch.mm(x,w11) + b11
  #out1 = (torch.mm(x,w11)
  #out1_final = torch.mm(out1,w21) + b21
  out1_final = torch.mm(out1,w21)
  return out1_final
  '''
  #out2 = F.tanh(torch.mm(x,w12) + b12)
  out2 = torch.mm(x,w12) + b12
  out2_final = torch.mm(out2,w22) + b22

  #out3 = F.tanh(torch.mm(x,w13) + b13)
  out3 = torch.mm(x,w13) + b13
  out3_final = torch.mm(out3,w23) + b23

  #out4 = F.tanh(torch.mm(x,w14) + b14)
  out4 = torch.mm(x,w14) + b14
  out4_final = torch.mm(out4,w24) + b24

  out = weight1*out1_final + weight2*out2_final + weight3*out3_final + weight4*out4_final
  return out
  '''



#mean = [0,0]
#cov = [[3,0],[0,1]]
#input_noise = np.random.multivariate_normal(mean, cov, n_data)
mean = 0
cov = 1
input_noise = np.random.normal(mean, cov, n_data)
#print('Input noise is', input_noise)
#syn_noise = 10000
#syn_noise = syn_noise.reshape(-1, n_dims)
ranged_data = np.random.rand(1) + 10000
ranged_data = torch.from_numpy(ranged_data).float()
ranged_data = ranged_data.reshape(-1, n_dims)
#print('Input noise is', input_noise.T)
input_noise = torch.from_numpy(input_noise).float()
input_noise = input_noise.reshape(-1, n_dims)

# Target dist
mean = 3
cov = 1
data = np.random.normal(mean, cov, n_data) # change name to target data
#data = np.arctan(data)
#print('Input noise is', input_noise.T)
data = torch.from_numpy(data).float()
data = data.reshape(-1, n_dims)


# create neural net model for sample generation
#model_sample_generation = NeuralNet(n_dims)
#original_data = model_sample_generation(input_noise)
#original_data = original_data.detach().numpy()
#data = torch.from_numpy(original_data).float()

criterion = nn.MSELoss()
#criterion = nn.KLDivLoss()
# TO DO : FIDDLE WITH THE PARAMETERS OF THE OPTIMIZER
#optimizer = torch.optim.Adam([w11, b11, w21, b21, w12, b12, w22, b22, w13, b13, w23, b23, w14, b14, w24, b24, weight1, weight2, weight3], lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
#optimizer = torch.optim.Adam([w11, b11, w21, b21, w12, b12, w22, b22, w13, b13, w23, b23, w14, b14, w24, b24, weight1, weight2, weight3], lr=learning_rate)
optimizer = torch.optim.Adam([w11, b11, w21, b21], lr=learning_rate)
#optimizer = torch.optim.Adam([w11, w21], lr=learning_rate)
for epoch in range(num_epochs):
    #for i, observations in enumerate(input_noise):
        # move the input_noise to the configured device
        #observations = input_noise.reshape(-1,n_dims)
        # forward pass
        optimizer.zero_grad()
        output = forward(input_noise)
        # loss traces the closeness of the samples generated
        loss = criterion(data,output)
        # backward pass
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print('Epoch', epoch, 'of', num_epochs, 'completed', 'with loss', loss)

with torch.no_grad():
    #for observations in input_noise_train:
    #observations = observations.reshape(-1,n_dims)
    output = forward(input_noise)
    #output_syn = forward(syn_noise)
    #print('Syn noise', output_syn)
    print('The weights of the first network are', w11, w21)
    print('The bias of the first network are', b11, b21)
    '''
    print('The weights of the second network are', w12, w22)
    print('The bias of the second network are', b12, b22)
    print('The weights of the third network are', w13, w23)
    print('The bias of the third network are', b13, b23)
    print('The weights of the fourth network are', w14, w24)
    print('The bias of the fourth network are', b14, b24)
    '''
    output_ranged = forward(ranged_data)
    output_ranged = output_ranged.detach().numpy()
    print('Output on ranged data is', output_ranged)
    #print(output)
    #print(input_noise)
    estimated_data = output.detach().numpy()
    data = data.detach().numpy()
    data_min = np.amin(data)
    #print(input_min)
    data_max = np.amax(data)
    #print(input_max)
    bins = np.linspace(data_min, data_max, 10)
    plt.hist(estimated_data, bins, alpha=0.5, label='estimated data')
    plt.hist(data, bins, alpha=0.5, label='original data')
    plt.legend(loc='upper right')
    plt.show()
