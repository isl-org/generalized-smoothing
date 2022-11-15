# Generalizing Gaussian Smoothing for Random Search

This repository contains code implementing the algorithms proposed in the paper [Generalizing Gaussian Smoothing for Random Search](https://proceedings.mlr.press/v162/gao22f.html), Gao and Sener (ICML 2022).

In particular, we provide the code used to obtain the experimental results on linear regression and the Nevergrad benchmark.
For online RL, we used the [ARS](https://github.com/modestyachts/ARS) repository; our proposed algorithms may be implemented by modifying the sampling distribution of the shared noise table.
Please see the paper for additional details and the hyperparameters used.

## Requirements

The code is written in Python 3.
Aside from the standard libraries, [NumPy](https://numpy.org/) and [Matplotlib](https://matplotlib.org/) are needed.
For linear regression, you also need [SciPy](https://scipy.org/), and for Nevergrad the corresponding [package](https://facebookresearch.github.io/nevergrad/).

## Running the experiments

Please see the READMEs in the `LinearRegression` and `benchmarks` folders for further instructions.

## Citation

To cite this repository in your research, please reference the following [paper]():

> Gao, Katelyn, and Ozan Sener. "Generalizing Gaussian Smoothing for Random Search." International Conference on Machine Learning. PMLR, 2022.

```TeX
@inproceedings{gao2022generalizing,
  title={Generalizing Gaussian Smoothing for Random Search},
  author={Gao, Katelyn and Sener, Ozan},
  booktitle={International Conference on Machine Learning},
  pages={7077--7101},
  year={2022},
  organization={PMLR}
}
```

## Contact

If you have questions, please contact <katelyn.gao@intel.com>.
