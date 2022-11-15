import argparse, os, json
import numpy as np

from models import Linear
from samplers import OnlineSampler, OfflineSampler
from worker import Worker
from algorithms import Perturbations
from optimizers import SGD

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--output", type=str, default='test')
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--model", type=str, default='random-multi')
    parser.add_argument("--algo", type=str, default='gs')
    parser.add_argument("--online", type=int, default=1)
    parser.add_argument("--antithetic", type=bool, default=False)
    parser.add_argument("--d", type=int, default=10)
    parser.add_argument("--loss", type=str, default='l2')
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--iters", type=int, default=1)

    parser.add_argument("--N", type=int, default = 30)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--L", type=int, default = 20)
    parser.add_argument("--c", type=float, default = 0.1)
    parser.add_argument("--lr", type=float, default = 0.01)

    ARGS = parser.parse_args()
    arg_dict = vars(ARGS)

    seed = arg_dict["seed"]
    np.random.seed(seed)
    exp_dir = os.getcwd()+'/'+arg_dict["output"]
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    json.dump(arg_dict, open(os.path.join(exp_dir, 'params.json'), 'w'), indent=2, sort_keys=True)

    # Instantiate objects for training and evaluation

    predictor = Linear(
            dim = arg_dict["d"],
            loss_type = arg_dict["loss"])

    online = arg_dict["online"]
    if online == 1:
        train_sampler = OnlineSampler(
                model = arg_dict["model"],
                dim = arg_dict["d"],
                num_data = arg_dict["N"])
    else:
        assert arg_dict["batch_size"] <= arg_dict["N"]
        train_sampler = OfflineSampler(
                model = arg_dict["model"],
                dim = arg_dict["d"],
                num_data = arg_dict["N"],
                batch_size = arg_dict["batch_size"])

    test_sampler = OfflineSampler(
            model = arg_dict["model"],
            dim = arg_dict["d"],
            num_data = 1000,
            batch_size = None)

    searcher = Perturbations(
            algo = arg_dict["algo"],
            dim = arg_dict["d"],
            L = arg_dict["L"])

    worker = Worker(
            predictor = predictor,
            train_sampler = train_sampler,
            test_sampler = test_sampler)

    optimizer = SGD(
            stepsize = arg_dict["lr"])

    # Alternate training and evaluation

    for r in range(arg_dict["rounds"]):

        with open(os.path.join(exp_dir, 'train.json'), 'a') as train_progress:

            for i in range(arg_dict["iters"]):
            
                # Generate perturbations
                epsilons = searcher.generate(num_perturbs = arg_dict["L"])

                # Compute pair of estimated risks at each perturbation 
                datum_losses = worker.loss(epsilons, arg_dict["c"], training = True, antithetic = arg_dict["antithetic"])
                losses = np.mean(datum_losses, axis=0)

                # Compute gradient estimate
                loss_diff = losses[:,0] - losses[:,1]
                grad = np.dot(np.transpose(epsilons), loss_diff)/(arg_dict["c"]*arg_dict["L"])
                if arg_dict["antithetic"]:
                    grad /= 2
                true_gradient = train_sampler.true_grad(predictor.params)
                mse = np.linalg.norm(grad-true_gradient)/np.sqrt(arg_dict["d"])

                # Optimizer step
                predictor.params = optimizer.update(predictor.params, grad)

                # Record statistics
                train_progress.write(json.dumps({
                        'round': r,
                        'iteration': i,
                        'loss': np.mean(losses),
                        'mse': mse,
                    }, cls=json.JSONEncoder))
                train_progress.write('\n')

        # Evaluation

        test_loss_est, test_loss_stderr = worker.loss(epsilons=None, c=None, training = False, antithetic = arg_dict["antithetic"])

        with open(os.path.join(exp_dir, 'test.json'), 'a') as test_progress:
            test_progress.write(json.dumps({
                'round': r,
                'loss_est': test_loss_est,
                'loss_stderr': test_loss_stderr/np.sqrt(1000),
                }, cls=json.JSONEncoder))
            test_progress.write('\n')
