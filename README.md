# Enforcing robust control guarantees within neural network policies

This repository is by 
[Priya L. Donti](https://www.priyadonti.com),
[Melrose Roderick](https://melroderick.github.io/),
[Mahyar Fazlyab](https://scholar.google.com/citations?user=Y3bmjJwAAAAJ&hl=en),
and [J. Zico Kolter](http://zicokolter.com),
and contains the [PyTorch](https://pytorch.org) source code to
reproduce the experiments in our paper
"[Enforcing robust control guarantees within neural network policies](https://arxiv.org/abs/2011.08105)."

If you find this repository helpful in your publications,
please consider citing our paper.

```
@inproceedings{donti2021enforcing,
  title={Enforcing robust control guarantees within neural network policies},
  author={Donti, Priya and Roderick, Melrose and Fazlyab, Mahyar and Kolter, J Zico},
  booktitle={International Conference on Learning Representations},
  year={2021}
}
```


## Introduction

When designing controllers for safety-critical systems, practitioners often face a challenging tradeoff between robustness and performance. While robust control methods provide rigorous guarantees on system stability under certain worst-case disturbances, they often result in simple controllers that perform poorly in the average (non-worst) case. In contrast, nonlinear control methods trained using deep learning have achieved state-of-the-art performance on many control tasks, but 
often lack robustness guarantees. We propose a technique that combines the strengths of these two approaches: a generic nonlinear control policy class, parameterized by neural networks, that nonetheless enforces the same provable robustness criteria as robust control. Specifically, we show that by integrating custom convex-optimization-based projection layers into a nonlinear policy, we can construct a provably robust neural network policy class that outperforms robust control methods in the average (non-adversarial) setting. We demonstrate the power of this approach on several domains, improving in performance over existing robust control methods and in stability over (non-robust) RL methods.

## Dependencies

+ Python 3.x/numpy/scipy/[cvxpy](http://www.cvxpy.org/en/latest/)
+ [PyTorch](https://pytorch.org) 1.5
+ OpenAI [Gym](https://gym.openai.com/) 0.15: *A toolkit for reinforcement learning*
+ [qpth](https://github.com/locuslab/qpth):
  *A fast differentiable QP solver for PyTorch*
+ [block](https://github.com/bamos/block):
  *A block matrix library for numpy and PyTorch*
+ [argparse](https://docs.python.org/3/library/argparse.html): *Input argument parsing*
+ [setproctitle](https://pypi.org/project/setproctitle/): *Library to set process titles*
+ [tqdm](https://tqdm.github.io/): *A library for smart progress bars*


## Instructions

### Running experiments

Experiments can be run the following commands for each environment (with the additional optional flag `--gpu [gpunum]` to enable GPU support). To reproduce the results in our paper, append the flag `--envRandomSeed 10` to the commands below.

Synthetic NLDI (D=0):

```
python main.py --env random_nldi-d0 
```
    
Synthetic NLDI (D ≠ 0):

```
python main.py --env random_nldi-dnonzero
```
    
Cart-pole:

```
python main.py --env cartpole --T 10 --dt 0.05
```
    
Planar quadrotor:

```
python main.py --env quadrotor --T 4 --dt 0.02
```

Microgrid:

```
python main.py --env microgrid
```

Synthetic PLDI:

```
python main.py --env random_pldi_env
```

Synthetic H<sub>∞</sub>:

```
python main.py --env random_hinf_env
```

### Generating plots

After running the experiments above, plots and tables can then be generated by running:

```
python plots.py
```
