'''
Contains functions to generate the perturbations for each algorithm.
'''

import numpy as np

class Perturbations(object):

    def __init__(self, algo, dim, L):

        self.algo_name = algo
        self.dim = dim
        self.L = L

        if algo == 'gs':
            self.dist = 'gaussian'
            self.sigma = 1
        elif algo == 'bes':
            self.dist = 'bernoulli'
            self.p = 0.5
            self.m = 0.5
        elif algo == 'gs-shrinkage':
            self.dist = 'gaussian'
            self.sigma = np.sqrt(L/(L+dim+1))
        elif algo == 'bes-shrinkage':
            self.dist = 'bernoulli'
            self.p = 0.5
            self.m = np.sqrt((L+dim-1)/(4*L))
        else:
            raise NotImplementedError

    def generate(self, num_perturbs):

        if self.dist == 'gaussian':
            return self.sigma * np.random.normal(size=(num_perturbs, self.dim))
        elif self.dist == 'bernoulli':
            return (np.random.binomial(1, self.p, size=(num_perturbs, self.dim))-self.p)/self.m
        else:
            raise NotImplementedError

