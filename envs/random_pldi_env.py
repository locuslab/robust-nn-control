import numpy as np
import torch
import torch.nn as nn

from envs import ode_env
import disturb_models as dm
from constants import *


class RandomPLDIEnv(ode_env.PLDIEnv):

    def __init__(self, n=5, m=3, L=3, random_seed=None, device=None):
        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed+1)

        self.n, self.m, self.L = n, m, L

        self.A = 3 * torch.tensor(np.random.randn(1, n, n) + 0.5 * np.random.randn(L, n, n), dtype=TORCH_DTYPE, device=device)
        self.B = 3 * torch.tensor(np.random.randn(1, n, m) + 0.5 * np.random.randn(L, n, m), dtype=TORCH_DTYPE, device=device)

        Q = np.random.randn(n, n)
        Q = Q.T @ Q
        self.Q = torch.tensor(Q, dtype=TORCH_DTYPE, device=device)

        R = np.random.randn(m, m)
        R = R.T @ R
        self.R = torch.tensor(R, dtype=TORCH_DTYPE, device=device)

        self.disturb_f = dm.PLDIDisturbModel(n, m, L)
        self.adversarial_disturb_f = None

        if device is not None:
            self.disturb_f.to(device=device, dtype=TORCH_DTYPE)

    def xdot_f(self, x, u, t):
        a = self.disturb_f(x, u, t)
        A = (self.A.unsqueeze(0) * a[:, :, None, None]).sum(1)
        B = (self.B.unsqueeze(0) * a[:, :, None, None]).sum(1)
        return (A @ x.unsqueeze(2) + B @ u.unsqueeze(2)).squeeze()

    def xdot_adversarial_f(self, x, u, t):
        if self.adversarial_disturb_f is None:
            raise ValueError('You must initialize adversarial_disturb_f before running in adversarial mode')
        a = self.adversarial_disturb_f(x, u, t)
        A = (self.A.unsqueeze(0) * a[:, :, None, None]).sum(1)
        B = (self.B.unsqueeze(0) * a[:, :, None, None]).sum(1)
        return (A @ x.unsqueeze(2) + B @ u.unsqueeze(2)).squeeze()

    def cost_f(self, x, u, t):
        return ((x @ self.Q) * x).sum(-1) + ((u @ self.R) * u).sum(-1)

    def get_pldi_linearization(self):
        return self.A, self.B, self.Q, self.R

    def gen_states(self, num_states, device=None):
        return torch.tensor(np.random.rand(num_states, self.n), device=device, dtype=TORCH_DTYPE)

    def __copy__(self):
        new_env = RandomPLDIEnv.__new__(RandomPLDIEnv)
        new_env.__dict__.update(self.__dict__)
        return new_env
