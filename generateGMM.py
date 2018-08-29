import numpy as np
import random
from random import randint
import math
import pdb
from scipy.stats import multivariate_normal


def generate_multivariate_GMM(n_data=100, n_dims=2, n_components=3, covariance_type = "diagonal", random_seed = 123):

    np.random.seed(random_seed) # set random seed

    # mean of the distributions
    mean = np.random.randint(20, size=(n_components, n_dims)) + np.random.rand(n_components, n_dims)
    # normalized random weights of the distributions
    weights = np.random.randint(20, size=(1, n_components))
    weights = weights/np.sum(weights)

    if covariance_type == "diagonal":
        sigma = np.random.randint(0,2,(n_components,n_dims)) + np.random.rand(n_components, n_dims)
        cov = np.zeros((n_components, n_dims, n_dims))
        for i in range(0,n_components):
            cov[i] = np.diag(sigma[i])
        print("Generating samples with diagonal covariance matrix")
    else:
        print("Generating samples with full covariance matrix")
        # covariance matrices of the distribution
        cov = np.random.randint(0, 2, (n_components, n_dims, n_dims)) + np.random.rand(n_components, n_dims, n_dims)
        # make the matrix positive semi-definite
        for i in range(0,n_components):
            cov[i] = np.dot(cov[i],cov[i].transpose())

    # Generate samples from the desired distribution
    data = []
    for i in range(0,n_components):
        component_data = np.random.multivariate_normal(mean[i], cov[i], math.ceil(weights[0, i]*n_data))
        data.extend(component_data)

    data = np.array(data[:n_data])
    np.random.shuffle(data)

    training_data = data[0:math.floor(0.5*n_data)]
    validation_data = data[math.ceil(0.5*n_data):math.floor(0.7*n_data)]
    test_data = data[math.ceil(0.7*n_data):]

    likelihood = np.zeros(np.size(test_data,0))

    for i in range(0,n_components):
        likelihood += weights[0,i]*multivariate_normal.pdf(test_data, mean[i], cov[i])
    avg_test_log_likelihood = sum(np.log(likelihood))/np.size(test_data,0)

    return data, training_data, validation_data, test_data, avg_test_log_likelihood

#n_data = 200
#n_dims = 2 # 16
#n_components = 4# 1024
#[data, training_data, validation_data, test_data, avg_test_log_likelihood] = generate_multivariate_GMM(n_data, n_dims, n_components)
#total_data = len(data)
