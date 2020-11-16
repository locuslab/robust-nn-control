import numpy as np
import torch
import torch.nn as nn

from envs import ode_env
import disturb_models as dm
from constants import *


class RandomHinfEnv(ode_env.HinfEnv):

    def __init__(self, n=5, m=3, wp=2, T=2, random_seed=None, device=None):
        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed+1)

        self.n, self.m, self.wp = n, m, wp

        self.A = torch.tensor(np.random.randn(n, n), dtype=TORCH_DTYPE, device=device)
        self.B = torch.tensor(np.random.randn(n, m), dtype=TORCH_DTYPE, device=device)
        self.G = torch.tensor(1.5*np.random.randn(n, wp), dtype=TORCH_DTYPE, device=device)

        Q = np.random.randn(n, n)
        Q = Q.T @ Q
        self.Q = torch.tensor(Q, dtype=TORCH_DTYPE, device=device)

        R = np.random.randn(m, m)
        R = R.T @ R
        self.R = torch.tensor(R, dtype=TORCH_DTYPE, device=device)

        self.disturb_f = dm.HinfDisturbModel(n, m, wp, T)
        self.adversarial_disturb_f = None

        if device is not None:
            self.disturb_f.to(device=device, dtype=TORCH_DTYPE)

    def xdot_f(self, x, u, t):
        w = self.disturb_f(x, u, t)
        return x @ self.A.T + u @ self.B.T + w @ self.G.T

    def xdot_adversarial_f(self, x, u, t):
        if self.adversarial_disturb_f is None:
            raise ValueError('You must initialize adversarial_disturb_f before running in adversarial mode')
        w = self.adversarial_disturb_f(x, u, t)
        return x @ self.A.T + u @ self.B.T + w @ self.G.T

    def cost_f(self, x, u, t):
        return ((x @ self.Q) * x).sum(-1) + ((u @ self.R) * u).sum(-1)

    def get_hinf_linearization(self):
        return self.A, self.B, self.G, self.Q, self.R

    def gen_states(self, num_states, device=None):
        return torch.randn((num_states, self.n), device=device, dtype=TORCH_DTYPE)

    def __copy__(self):
        new_env = RandomHinfEnv.__new__(RandomHinfEnv)
        new_env.__dict__.update(self.__dict__)
        return new_env
