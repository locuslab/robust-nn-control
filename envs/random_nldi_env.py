import numpy as np
import torch
import torch.nn as nn

from envs import ode_env
import disturb_models as dm
from constants import *


class RandomNLDIEnv(ode_env.NLDIEnv):

    def __init__(self, n=5, m=3, wp=2, wq=2, isD0=False, random_seed=None, device=None):
        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed+1)

        self.n, self.m, self.wp, self.wq = n, m, wp, wq
        self.isD0 = isD0

        self.A = torch.tensor(np.random.randn(n, n), dtype=TORCH_DTYPE, device=device)
        self.B = torch.tensor(np.random.randn(n, m), dtype=TORCH_DTYPE, device=device)
        self.G = torch.tensor(1.5*np.random.randn(n, wp), dtype=TORCH_DTYPE, device=device)
        self.C = torch.tensor(np.random.randn(wq, n), dtype=TORCH_DTYPE, device=device)
        if isD0:
            self.D = torch.zeros(wq, m, dtype=TORCH_DTYPE, device=device)
        else:
            self.D = torch.tensor(0.01*np.random.randn(wq, m), dtype=TORCH_DTYPE, device=device)

        Q = np.random.randn(n, n)
        Q = Q.T @ Q
        self.Q = torch.tensor(Q, dtype=TORCH_DTYPE, device=device)

        R = np.random.randn(m, m)
        R = R.T @ R
        self.R = torch.tensor(R, dtype=TORCH_DTYPE, device=device)

        self.disturb_f = dm.NLDIDisturbModel(self.C, self.D, n, m, wp)
        self.adversarial_disturb_f = None

        if device is not None:
            self.disturb_f.to(device=device, dtype=TORCH_DTYPE)

    def xdot_f(self, x, u, t):
        p = self.disturb_f(x, u, t)
        return x @ self.A.T + u @ self.B.T + p @ self.G.T

    def xdot_adversarial_f(self, x, u, t):
        if self.adversarial_disturb_f is None:
            raise ValueError('You must initialize adversarial_disturb_f before running in adversarial mode')
        p = self.adversarial_disturb_f(x, u, t)
        return x @ self.A.T + u @ self.B.T + p @ self.G.T

    def cost_f(self, x, u, t):
        return ((x @ self.Q) * x).sum(-1) + ((u @ self.R) * u).sum(-1)

    def get_nldi_linearization(self):
        return self.A, self.B, self.G, self.C, self.D, self.Q, self.R

    def gen_states(self, num_states, device=None):
        return torch.tensor(np.random.rand(num_states, self.n), device=device, dtype=TORCH_DTYPE)

    def __copy__(self):
        new_env = RandomNLDIEnv.__new__(RandomNLDIEnv)
        new_env.__dict__.update(self.__dict__)
        return new_env

