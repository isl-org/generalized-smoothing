## Reproducing the linear regression results

In this folder we provide code to reproduce our online linear regression experiments.
We give command line instructions; please refer to the supplement of our paper for the hyperparameters used.

### From the terminal

```
python runner.py --output [out] --seed [s] --algo [algorithm] --d [dim] --rounds 100 --iters 10 
    --N [N] --L [L] --c [c] --lr [lr]
```
The inputs are:
* `[out]`: folder where you'd like to put the output files
* `[s]`: random seed for NumPy generator
* `[algorithm]`: name of the algorithm; _gs_, _bes_, _gs-shrinkage_, or _bes-shrinkage_
* `[dim]`: problem dimension
* `[N]`: number of data points generated and used to estimate losses at each iteration
* `[L]`: number of perturbations sampled at each iteration
* `[c]`: spacing hyperparameter in the gradient estimator
* `[lr]`: learning rate for gradient descent
* There are 100 rounds, each with 10 optimization iterations

Currently we use the forward difference gradient estimator.
To use the antithetic gradient estimator, modify the corresponding parser argument default value in `runner.py`.
Offline linear regression can also be run, where `[N]` is the dataset size.
Just modify the `--online` and `--batch_size` parser argument default values in `runner.py`.

Each run of the above command creates three files in the folder `[out]`: 1) `params.json` containing the inputs 2) `train.json` containing the mean squared error of the gradient estimate at each iteration and 3) `test.json` containing the loss on the test dataset at each round.

### Interpreting the output

Suppose that we have run the above command for different sets of inputs, each with the same fixed set of random seeds.
To plot the average test loss at each round for each set of inputs, from the terminal run
```
python comparison_plots.py --output [out] --num_comps [comps] --seeds [seeds] --folder0 [folder0] 
    --folder1 [folder1] ... --folder5 [folder5]
```
* `[out]`: folder where you'd like to put the plot
* `[comps]`: number of sets of inputs you'd like to compare the performance of, up to six
* `[seeds]`: the seeds with which `runner.py` were run
* `[folder0]`: folder where the results for the first set of inputs are stored, consisting of subfolders `seed_s` for each random seed `s`; each subfolder contains the output files from executing `runner.py`
* `[folder1]`, ..., `[folder5]`: folders where the results for the second, ..., sixth set of inputs are stored

In the folder `[out]`, there are: 1) `test.png`, the plot of the test loss 2) `train_mse.png`, the plot of the mean squared error of the gradient estimate during training 3) `output.txt`, recording the final average performance for each set of inputs and 4) `members.txt`, recording `[folder0]`, ..., `[folder5]` and their corresponding colors in the plot.
