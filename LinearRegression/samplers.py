'''
Classes and methods to sample test data and data at each training iteration.
'''

import numpy as np
from scipy.stats import special_ortho_group

def generate_task_random(dim, beta=2):

    initializer = np.random.uniform(0, 1, dim)
    Theta = beta*initializer

    if dim == 1:
        U = np.ones((1,1))
    else:
        U = special_ortho_group.rvs(dim)
    Lambda = beta*np.diag(initializer)
    Q = np.matmul(U, np.matmul(Lambda, np.transpose(U)))

    sigma = np.random.uniform(0, beta)

    return Q, Theta, sigma

def true_erm_grad(model, dim, mc=1000):

    '''
    Function to compute the ground truth gradient, assuming L2 loss
    and the data distribution described in the paper (random-multi)
    '''

    EQT = np.zeros((dim, ))
    if model == "random-multi":
        for k in range(mc):
            Q, Theta, sigma = generate_task_random(dim)
            EQT += np.dot(Q, Theta)/mc
    else:
        raise NotImplementedError

    def true_grad(thetahat):
        return thetahat-EQT

    return true_grad

def generate_data(model, dim, num_data):

    features = np.zeros((num_data, dim))
    responses = np.zeros((num_data,))

    for j in range(num_data):
        if model == "random-multi":
            Q, Theta, sigma = generate_task_random(dim)
        else:
            raise NotImplementedError
        features[j,:] = np.random.multivariate_normal(np.zeros(dim), Q)
        responses[j] = np.dot(features[j,:], Theta)+np.random.normal(0, np.sqrt(sigma))

    return features, responses

class OnlineSampler(object):

    def __init__(self, model, dim, num_data):

        self.model = model
        self.dim = dim
        self.num_data = num_data

        self.true_grad = true_erm_grad(model, dim)

    def sample(self):

        features, responses = generate_data(self.model, self.dim, self.num_data)
        return features, responses

class OfflineSampler(object):
    
    def __init__(self, model, dim, num_data, batch_size):

        self.model = model
        self.dim = dim
        self.num_data = num_data
        self.batch_size = batch_size

        self.features, self.responses = generate_data(model, dim, num_data)
        self.true_grad = true_erm_grad(model, dim)

    def data(self):

        return self.features, self.responses

    def sample(self):

        idxs = np.random.choice(self.num_data, self.batch_size, replace=False)
        features = self.features[idxs, :]
        responses = self.responses[idxs]
        return features, responses


