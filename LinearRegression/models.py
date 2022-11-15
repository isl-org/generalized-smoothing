'''
Object classes for models.
Contains methods to compute predictions and losses on a batch of data.
'''

import numpy as np

class Model(object):
    
    def __init__(self, dim, loss_type):

        self.dim = dim
        self.loss_type = loss_type
    
    def predict(self, X, params=None):

        pass

    def loss(self, targets, predictions):

        pass

class Linear(Model):

    def __init__(self, dim, loss_type):

        Model.__init__(self, dim, loss_type)
        self.params = np.random.normal(size=dim)

    def predict(self, X, params=None):

        if params is None:
            params = self.params
            
        return np.dot(X, params)

    def loss(self, targets, predictions):

        if self.loss_type == 'l2':
            errors = 0.5*np.square(targets-predictions)
            self.average_loss = np.mean(errors)
            self.loss_std = np.std(errors)
        else:
            raise NotImplementedError

        return self.average_loss, self.loss_std
