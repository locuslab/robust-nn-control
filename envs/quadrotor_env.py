import numpy as np
import torch

from envs import ode_env
import disturb_models as dm
from constants import *

import os


class QuadrotorEnv(ode_env.NLDIEnv):

    def __init__(self, mass=0.8, moment_arm=0.01, inertia_roll=15.67e-3, random_seed=None, device=None):
        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed+1)
            
        self.device = device

        # guessing parameters:
        #    mass 1 kg, moment of inertia for roll axis 0.0093 kg * m^2 
        #       http://www.diva-portal.org/smash/get/diva2:1020192/FULLTEXT02.pdf
        #    OR mass 0.8 kg, moment of inertia for roll axis 15.67e-3 kg*m^2, arm length of vehicle=0.3m
        #       https://www.researchgate.net/publication/283241371_Feedback_control_strategies_for_quadrotor-type_aerial_robots_a_survey

        self.g = 9.81  # gravitational acceleration in m/s^2

        self.mass = mass
        self.moment_arm = moment_arm
        self.inertia_roll = inertia_roll

        # Max and min values for the state: [x, z, roll, xdot, zdot, rolldot]
        self.x_max = torch.tensor([1.1, 1.1, 0.06, 0.5, 1.0, 0.8], dtype=TORCH_DTYPE, device=device)
        self.x_min = torch.tensor([-1.1, -1.1, -0.06, -0.5, -1.0, -0.8], dtype=TORCH_DTYPE, device=device)

        self.x_0_max = torch.tensor([1.0, 1.0, 0.05, 0.0, 0.0, 0.0], dtype=TORCH_DTYPE, device=device)
        self.x_0_min = torch.tensor([-1.0, -1.0, -0.05, -0.0, -0.0, -0.0], dtype=TORCH_DTYPE, device=device)

        self.B = torch.tensor([
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [1/self.mass, 1/self.mass],
                [self.moment_arm/self.inertia_roll, -self.moment_arm/self.inertia_roll]
            ], dtype=TORCH_DTYPE, device=device)

        # TODO: hacky, assumes call from main.py in top level directory
        array_path = os.path.join('problem_gen', 'quadrotor')
        self.A = torch.tensor(np.load(os.path.join(array_path, 'A.npy')), dtype=TORCH_DTYPE, device=device)
        self.n, self.m = self.A.shape[0], self.B.shape[1]

        self.G_lin = torch.tensor(np.load(os.path.join(array_path, 'G.npy')), dtype=TORCH_DTYPE, device=device)
        self.C_lin = torch.tensor(np.load(os.path.join(array_path, 'C.npy')), dtype=TORCH_DTYPE, device=device)
        self.D_lin = torch.zeros(self.C_lin.shape[0], self.m, dtype=TORCH_DTYPE, device=device)

        disturb_n = 2
        self.G_disturb = 0.1 * torch.tensor(np.random.randn(self.n, disturb_n), dtype=TORCH_DTYPE, device=device)
        self.C_disturb = torch.tensor(0.1 * np.random.randn(disturb_n, self.n), dtype=TORCH_DTYPE, device=device)
        self.D_disturb = torch.zeros(disturb_n, self.m, dtype=TORCH_DTYPE, device=device)

        self.G = torch.cat([self.G_lin, self.G_disturb], dim=1)
        self.C = torch.cat([self.C_lin, self.C_disturb], dim=0)
        self.D = torch.cat([self.D_lin, self.D_disturb], dim=0)

        self.wp, self.wq = self.G.shape[1], self.C.shape[0]

        # TODO: have reasonable objective?
        Q = np.random.randn(self.n, self.n)
        Q = Q.T @ Q
        # Q = np.eye(self.n)
        self.Q = torch.tensor(Q, dtype=TORCH_DTYPE, device=device)

        R = np.random.randn(self.m, self.m)
        R = R.T @ R
        # R = np.eye(self.m)
        self.R = torch.tensor(R, dtype=TORCH_DTYPE, device=device)

        self.disturb_f = dm.NLDIDisturbModel(self.C_disturb, self.D_disturb, self.n, self.m, self.G_disturb.shape[1])
        if device is not None:
            self.disturb_f.to(device=device, dtype=TORCH_DTYPE)

        self.adversarial_disturb_f = None

    def xdot_f(self, x, u, t):
        px, pz, phi, vx, vz, phidot = [x[:,i] for i in range(x.shape[1])]

        x_part = torch.stack([
            vx*torch.cos(phi) - vz*torch.sin(phi), 
            vx*torch.sin(phi) + vz*torch.cos(phi),
            phidot,
            vz*phidot - self.g*torch.sin(phi),
            -vx*phidot - self.g*torch.cos(phi) + self.g,   # note: + g = center dynamics by assuming nominal policy [gm/2, gm/2] always applied
            torch.zeros(x.shape[0], device=self.device, dtype=TORCH_DTYPE)
        ]).T

        p_disturb = self.disturb_f(x, u, t)
        return x_part + u@self.B.T + p_disturb @ self.G_disturb.T
    
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
        prop = torch.tensor(np.random.rand(num_states, self.n), device=device, dtype=TORCH_DTYPE)
        return self.x_0_max.detach()*prop + self.x_0_min.detach()*(1-prop)

    def __copy__(self):
        new_env = QuadrotorEnv.__new__(QuadrotorEnv)
        new_env.__dict__.update(self.__dict__)
        return new_env


