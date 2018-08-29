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
hidden_layer1 = 3
hidden_layer2 = 4
hidden_layer3 = 3
output_layer = 2
# hyperparameters for data generation
n_dims = 2
n_components = 1
n_data = 10
# hyperparameters for network training
num_epochs = 1000
batch_size = 100
learning_rate = 0.01

# Network architecture for data generation
class NeuralNet_Samples(nn.Module):
    def __init__(self, n_dims):
        super(NeuralNet_Samples, self).__init__()
        self.transformation_samples = nn.Sequential(
                    nn.Linear(n_dims, 4),
                    nn.ReLU(),
                    nn.Linear(4, 2),
                    nn.Tanh(),
                    nn.Linear(2, 4),
                    nn.Sigmoid(),
                    nn.Linear(4, n_dims),
                    nn.Softplus())
    def forward(self, x):
        #encoded = self.encoder(x)
        out = self.transformation_samples(x)
        return out
# Network architecture for training
class NeuralNet_learn(nn.Module):
    def __init__(self, n_dims):
        super(NeuralNet_learn, self).__init__()
        self.transformation = nn.Sequential(
                    nn.Linear(n_dims, 2),
                    nn.ReLU(),
                    nn.Linear(2, 4),
                    nn.Tanh(),
                    nn.Linear(4, 2),
                    nn.ReLU(),
                    nn.Linear(2, n_dims),
                    nn.Softplus())
    def forward(self, x):
        #encoded = self.encoder(x)
        out = self.transformation(x)
        return out

'''
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_layer1, hidden_layer2, hidden_layer3, output_layer):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_layer1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_layer1, hidden_layer2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_layer2, hidden_layer3)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_layer3, output_layer)
        self.relu4 = nn.ReLU()


    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        out = self.relu4(out)
        return out
'''

# generate sampes from a target distribution by feeding input noise through a neural network
[input_noise, training_data, validation_data, test_data, avg_test_log_likelihood] = generate_multivariate_GMM(n_data, n_dims, n_components, covariance_type="full", random_seed = np.random.randint(1,500))
# arrange the input_noise in proper format for neural network
'''
input_noise = torch.from_numpy(input_noise).float()
input_noise = input_noise.reshape(-1, n_dims)
# create the neural net model
model_sample_generation = NeuralNet_Samples(n_dims)
original_data = model_sample_generation(input_noise)
original_data = original_data.detach().numpy()
data = torch.from_numpy(original_data).float()
'''
data = torch.from_numpy(input_noise).float()
#model = NeuralNet(input_size, hidden_layer1, hidden_layer2, hidden_layer3, output_layer)

# create neural network model for training
model = NeuralNet_learn(n_dims)

# TO DO : GENERATE GAUSSIAN NOISE WITH IDENTITY COVARIANCE IN n_dims HERE!!!
'''
[input_noise_train, _, _, _, _] = generate_multivariate_GMM(n_data, n_dims, 1, covariance_type="full", random_seed = np.random.randint(1,500))
input_noise_train = torch.from_numpy(input_noise_train).float()
input_noise_train = input_noise_train.reshape(-1, n_dims)
'''
input_noise = input_noise.reshape(-1, n_dims)
#input_noise_train = input_noise
'''
#input_noise = np.random.multivariate_normal(mean, cov, n_data)
input_noise = torch.from_numpy(input_noise).float()
input_noise = input_noise.reshape(-1,n_dims)
original_data = model(input_noise)
original_data = original_data.detach().numpy()
data = torch.from_numpy(original_data).float()
'''
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
        observations = input_noise.reshape(-1,n_dims)
        # forward pass
        output = model(observations)
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
    observations = observations.reshape(-1,n_dims)
    output = model(observations)
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
    #sns.kdeplot(xaxis, original_data, bw= 2, label="Original data")
    #sns.kdeplot(xaxis, estimated_data, bw=2, label="Estimated data")
