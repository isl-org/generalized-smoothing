'''
Given folders containing runs of algorithms on a benchmark over a set of random seeds, 
plots their oracle performances on the same graph (test.png)
and records their corresponding final values (output.txt).
'''

import matplotlib.pyplot as plt
import numpy as np
import argparse
import os, json

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--output', type=str, default='test')
    parser.add_argument('--num_comps', type=int, default=6)
    parser.add_argument('--seeds', type=str, default='1,2,3,4,5')
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--folder0', type=str)
    parser.add_argument('--folder1', type=str)
    parser.add_argument('--folder2', type=str)
    parser.add_argument('--folder3', type=str)
    parser.add_argument('--folder4', type=str)
    parser.add_argument('--folder5', type=str)

    args = parser.parse_args()
    arg_dict = vars(args)

    outdir = arg_dict["output"]
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    seeds = [int(item) for item in arg_dict["seeds"].split(',')]
    colors = ['tab:red', 'k', 'tab:orange', 'tab:blue', 'tab:purple', 'tab:brown']

    plt.rc('axes', titlesize=16)
    plt.rc('axes', labelsize=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    f1 = plt.figure()
    ax1 = f1.gca()
    f2 = plt.figure()
    ax2 = f2.gca()

    for i in range(arg_dict["num_comps"]):
        job_dir = arg_dict["folder"+str(i)]
        with open(os.path.join(outdir, 'members.txt'), 'a') as record:
            record.write(job_dir+' '+colors[i])
            record.write('\n')

        test_mean = [[] for r in range(arg_dict["iters"])]
        test_stderr = [[] for r in range(arg_dict["iters"])]
        train_mse = [[] for r in range(arg_dict["iters"])]
        for s in range(len(seeds)):
            log_dir = os.path.join(job_dir, 'seed_'+str(seeds[s]))
            training_file = os.path.join(log_dir, 'train.json')
            evaluation_file = os.path.join(log_dir, 'test.json')

            # Testing results
            results = []
            with open(evaluation_file, 'r') as test:
                for line in test:
                    results.append(json.loads(line))
            for data in results:
                r = data['round']
                test_mean[r].append(data['loss_est'])
                test_stderr[r].append(data['loss_stderr'])

            # Training results
            train_obj = []
            with open(training_file, 'r') as train:
                for line in train:
                    train_obj.append(json.loads(line))
            for data in train_obj:
                r = data['round']
                train_mse[r].append(data['mse'])

        compiled_rounds = np.arange(arg_dict["iters"])
        compiled_test_mean = np.asarray([np.mean(np.asarray(test_mean[r])) for r in range(arg_dict["iters"])])
        compiled_test_var = np.asarray([(np.var(np.asarray(test_mean[r]))
            +np.mean(np.asarray(test_stderr[r])**2))/len(test_mean[r]) for r in range(arg_dict["iters"])])
        compiled_train_mse = np.asarray([np.mean(np.asarray(train_mse[r])) for r in range(arg_dict["iters"])])
        compiled_mse_stderr = np.asarray([np.std(np.asarray(train_mse[r]))/np.sqrt(10*len(seeds)) for r in range(arg_dict["iters"])])
        with open(os.path.join(outdir, 'output.txt'), 'a') as printout:
            printout.write('For folder {}, the final test performance is {} with standard error {}'.format(i, compiled_test_mean[-1], np.sqrt(compiled_test_var[-1])))
            printout.write('\n')

        ax1.plot(compiled_rounds, compiled_test_mean, colors[i])
        ax1.fill_between(compiled_rounds, compiled_test_mean-np.sqrt(compiled_test_var),
                compiled_test_mean+np.sqrt(compiled_test_var), facecolor=colors[i], alpha=0.25)
        ax2.plot(compiled_rounds, compiled_train_mse, colors[i])
        ax2.fill_between(compiled_rounds, compiled_train_mse-compiled_mse_stderr,
                compiled_train_mse+compiled_mse_stderr, facecolor=colors[i], alpha=0.25)

    ax1.set_xlabel('Round')
    ax1.set_ylabel('Test loss')
    f1.savefig(os.path.join(outdir, 'test.png'), bbox_inches = 'tight')
    plt.close(f1)

    ax2.set_xlabel('Round')
    ax2.set_ylabel('Gradient MSE')
    f2.savefig(os.path.join(outdir, 'train_mse.png'), bbox_inches = 'tight')
    plt.close(f2)

