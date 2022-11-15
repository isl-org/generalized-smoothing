import argparse, os, json
import numpy as np

from optimizers import SGD, SGD_history
from algorithms import Perturbations
from nevergrad.functions.functionlib import ArtificialFunction
from utils import gram_schmidt

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--output", type=str, default='test')
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--objective", type=str, default='sphere')
    parser.add_argument("--algo", type=str, default='gs')

    parser.add_argument("--antithetic", type=bool, default=False)
    parser.add_argument("--d", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--iters", type=int, default=1)

    parser.add_argument("--N", type=int, default = 1)
    parser.add_argument("--L", type=int, default=20)
    parser.add_argument("--c", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.001)

    ARGS = parser.parse_args()
    arg_dict = vars(ARGS)

    seed = arg_dict["seed"]
    np.random.seed(seed)
    exp_dir = os.getcwd()+'/'+arg_dict["output"]
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    json.dump(arg_dict, open(os.path.join(exp_dir, 'params.json'), 'w'), indent=2, sort_keys=True)

    d = arg_dict["d"]
    theta = np.random.normal(size=d)

    name = arg_dict["objective"]
    obj = ArtificialFunction(name, d, noise_level=0.1, translation_factor=0.0)
    oracle = ArtificialFunction(name, d, noise_level=0.0, translation_factor=0.0)

    searcher = Perturbations(
            algo = arg_dict["algo"],
            dim = d,
            L = arg_dict["L"])

    if arg_dict["algo"] == "guided-es":
        optimizer = SGD_history(stepsize=arg_dict["lr"], maxlen = 50)
    else:
        optimizer = SGD(stepsize = arg_dict["lr"])

    # Alternate training and evaluation

    for r in range(arg_dict["rounds"]):
        for i in range(arg_dict["iters"]):

            # Generate perturbations
            epsilons = searcher.generate(num_perturbs = arg_dict["L"])
            if arg_dict["algo"] == 'orthogonal-es':
                epsilons, _ = gram_schmidt(epsilons, 'row')
            elif arg_dict["algo"]== 'guided-es':
                if (r*arg_dict["iters"]+i)>=optimizer.maxlen:
                    _, basis = gram_schmidt(optimizer.get_history(), 'row')
                    new_cov = 0.5*np.identity(d)/d
                    new_cov += 0.5*np.matmul(np.transpose(basis), basis)/optimizer.maxlen
                else:
                    new_cov = np.identity(d)/d
                epsilons = np.transpose(np.matmul(np.linalg.cholesky(new_cov), np.transpose(epsilons)))

            # Compute gradient estimates
            grad_est = np.zeros(d)
            for l in range(arg_dict["L"]):
                epsilon = epsilons[l]
                loss1 = 0
                for n in range(arg_dict["N"]):
                    loss1 += obj(theta+arg_dict["c"]*epsilon)/arg_dict["N"]
                loss2 = 0
                if arg_dict["antithetic"]:
                    for n in range(arg_dict["N"]):
                        loss2 += obj(theta-arg_dict["c"]*epsilon)/arg_dict["N"]
                    grad_est += (loss1-loss2)*epsilon/(2*arg_dict["c"]*arg_dict["L"])
                else:
                    for n in range(arg_dict["N"]):
                        loss2 += obj(theta)/arg_dict["N"]
                    grad_est += (loss1-loss2)*epsilon/(arg_dict["c"]*arg_dict["L"])

            # Optimizer step
            theta -= optimizer.step(grad_est)

        # Evaluation
        with open(os.path.join(exp_dir, 'test.json'), 'a') as test_progress:
            test_progress.write(json.dumps({
                'round': r,
                'loss': oracle(theta),
                }, cls=json.JSONEncoder))
            test_progress.write('\n')
