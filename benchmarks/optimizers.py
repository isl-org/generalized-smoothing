"""
Methods to compute the descent steps given a gradient estimate.
SGD_history is only used for Guided ES, which requires storing the last k gradient estimates
in order to compute the distribution of the perturbations.
"""

from collections import deque
import numpy as np

class SGD(object):

    def __init__(self, stepsize):
        self.stepsize = stepsize

    def step(self, gradient):
        return self.stepsize*gradient

class SGD_history(SGD):

    def __init__(self, stepsize, maxlen):

        SGD.__init__(self, stepsize)
        self.maxlen = maxlen
        self.history = deque()

    def step(self, gradient):

        # Update history
        if len(self.history) == self.maxlen:
            self.history.popleft()
        self.history.append(gradient)

        return self.stepsize*gradient

    def get_history(self):

        li = list(self.history)

        return np.asarray(li)
