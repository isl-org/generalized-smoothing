'''
Methods to update parameters with gradient descent given a gradient estimate.
'''

class SGD(object):

    def __init__(self, stepsize):

        self.stepsize = stepsize

    def update(self, variable, gradient):

        return variable - self.stepsize*gradient
