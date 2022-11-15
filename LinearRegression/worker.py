'''
Compute, at each training iteration, the losses of the perturbed parameters on each data point
and the average loss at test time
'''

import numpy as np

class Worker(object):

    def __init__(self, predictor, train_sampler, test_sampler):

        self.predictor = predictor
        self.train_sampler = train_sampler
        self.test_sampler = test_sampler

    def loss(self, epsilons, c, training, antithetic):

        if training:
            features, responses = self.train_sampler.sample()
            detailed_losses = np.zeros((responses.size, epsilons.shape[0], 2))

            if antithetic:
                for l in range(epsilons.shape[0]):
                    pos_preds = self.predictor.predict(features, self.predictor.params+c*epsilons[l,:])
                    neg_preds = self.predictor.predict(features, self.predictor.params-c*epsilons[l,:])
                    for m in range(responses.size):
                        detailed_losses[m,l,0] = self.predictor.loss(responses[m], pos_preds[m])[0]
                        detailed_losses[m,l,1] = self.predictor.loss(responses[m], neg_preds[m])[0]
            else:
                # At current parameters
                current_preds = self.predictor.predict(features)

                # At perturbed parameters
                for l in range(epsilons.shape[0]):
                    pos_preds = self.predictor.predict(features, self.predictor.params+c*epsilons[l,:])
                    for m in range(responses.size):
                        detailed_losses[m,l,0] = self.predictor.loss(responses[m], pos_preds[m])[0]
                        detailed_losses[m,l,1] = self.predictor.loss(responses[m], current_preds[m])[0]
            
            return detailed_losses
        else:
            features, responses = self.test_sampler.data()
            preds = self.predictor.predict(features)
            return self.predictor.loss(responses, preds)
