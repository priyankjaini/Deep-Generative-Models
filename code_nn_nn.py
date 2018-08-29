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

# hyperparameters for network architecture
input_size = 2
hidden_layer1 = 4
hidden_layer2 = 8
hidden_layer3 = 4
output_layer = 2

# hyperparameters for data generation
n_dims = 2
n_components = 1
n_data = 100
# hyperparameters for network training
num_epochs = 10000
batch_size = 100
learning_rate = 0.01
# Neural network architecture for both sample generation and training
class NeuralNet(nn.Module):
    def __init__(self, n_dims):
        super(NeuralNet, self).__init__()
        self.transformation = nn.Sequential(
                    nn.Linear(n_dims, hidden_layer1),
                    nn.ReLU(),
                    nn.Linear(hidden_layer1, hidden_layer2),
                    nn.Tanh(),
                    nn.Linear(hidden_layer2, hidden_layer3),
                    nn.Sigmoid(),
                    nn.Linear(hidden_layer3, n_dims),
                    nn.Softplus())
    def forward(self, x):
        out = self.transformation(x)
        return out
#[input_noise, _, _, _, _] = generate_multivariate_GMM(n_data, n_dims, n_components, covariance_type="full", random_seed = np.random.randint(1,500))
mean = [0,0]
cov = [[3,0],[0,1]]
input_noise = np.random.multivariate_normal(mean, cov, n_data)
#input_noise = np.random.normal(mean, cov, n_data)
#print('Input noise is', input_noise.T)
input_noise = torch.from_numpy(input_noise).float()
input_noise = input_noise.reshape(-1, n_dims)
# create neural net model for sample generation
model_sample_generation = NeuralNet(n_dims)
original_data = model_sample_generation(input_noise)
original_data = original_data.detach().numpy()
data = torch.from_numpy(original_data).float()
# create neural network model for training
model = NeuralNet(n_dims)

# Loss and Optimizer
# TO DO : CODE RECONSTRUCTION LOSS AND USE THAT
# MSE loss
#criterion = nn.MSELoss()
# KL divergence
criterion = nn.KLDivLoss()
# TO DO : FIDDLE WITH THE PARAMETERS OF THE OPTIMIZER
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

# Train the model
# TO DO : INCORPORATE BATCH LEARNING HERE
for epoch in range(num_epochs):
    #for i, observations in enumerate(input_noise):
        # move the input_noise to the configured device
        #observations = input_noise.reshape(-1,n_dims)
        # forward pass
        output = model(input_noise)
        # loss traces the closeness of the samples generated
        loss = criterion(data,output)
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #if (i+1) % 10 == 0:
            #print ('Epoch [{}/{}], Loss: {:.4f}'
                    #.format(epoch+1, num_epochs, i+1, loss.item()))

# test the model
# use no gradients because we don't need to trace gradients anymore
with torch.no_grad():
    #for observations in input_noise_train:
    #observations = observations.reshape(-1,n_dims)
    output = model(input_noise)
    print(output)
    print(data)
    estimated_data = output.detach().numpy()
'''
    # plotting the target and estimated distribution
    input_min = np.amin(input_noise)
    #print(input_min)
    input_max = np.amax(input_noise)
    #print(input_max)
    #xa = np.linspace(-1, 1, 0.05)
    #sns.kdeplot(original_data)
    #plt.plot(original_data[:,0], original_data[:,1],'r--',estimated_data[:,0], estimated_data[:,1],'bs')
    #plt.hist(input_noise)
    #plt.show()
    #plt.hist(estimated_data)
    #plt.show()
    bins = np.linspace(input_min, input_max, 30)
    plt.hist(estimated_data, bins, alpha=0.5, label='estimated data')
    plt.hist(input_noise, bins, alpha=0.5, label='original data')
    plt.legend(loc='upper right')
    plt.show()
    #plt.plot(original_data)
'''
