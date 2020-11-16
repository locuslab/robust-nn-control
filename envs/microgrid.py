import numpy as np
import torch

from envs import ode_env
import disturb_models as dm
from constants import *

'''
Adapted from:
Quang Linh Lam, Antoneta Iuliana Bratcu, Delphine Riu. Frequency robust control in stand-alone
microgrids with PV sources : design and sensitivity analysis. Symposium de Génie Electrique, Jun
2016, Grenoble, France. ffhal-01361556
'''

class MicrogridEnv(ode_env.NLDIEnv):

    def __init__(self, random_seed=None, device=None):
        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed+1)

        self.n = 3
        self.m = 2
        self.wp = 1
        self.wq = 1
        self.isD0 = True   # TODO?

        # for per-unit normalization
        v_base = 585
        p_base = 1.0015
        i_base = p_base / v_base
        r_base = v_base / i_base

        omega_b = 3.5  # between 2.61 and 5.22 given in paper
        R_dc = 50 / r_base   # based on 5-200 ohms range at https://doc.ingeniamc.com/venus/product-manual/installation-and-configuration/motor-output-wiring/shunt-braking-resistor
        C_dc = 500e-6 * r_base     # based on 200-800 micro-farads given here: https://www.elmomc.com/wp-content/uploads/2019/08/Simple-Capacitor-white-paper.pdf
                        # and also approx here: https://doc.ingeniamc.com/titan/manuals/titan-go-product-manual/wiring-and-connections/dc-bus-bulk-capacitance
        T_diesel = 0.01  # based on plausible value from https://core.ac.uk/download/pdf/52115363.pdf (TODO switch to p.u.?)
        s_diesel = 0.04  # arbitrary, based on plausible value from Wikipedia: https://en.wikipedia.org/wiki/Droop_speed_control
        H = 0.7305  # based on plausible value from https://core.ac.uk/download/pdf/52115363.pdf (TODO switch to p.u.?)
        D_load = 0.9   # arbitrary
        alpha_ce = 0.585
        beta_de = 0.4
        v_sce = 585 / v_base
        R_sc = 30e-3 / r_base    # based on plausible internal supercapacitor resistence of 30 milli-ohms from https://en.wikipedia.org/wiki/Supercapacitor#Internal_resistance
        i_se = -2.5 / i_base

        self.A = 0.001 * torch.tensor((
            (-omega_b/(R_dc*C_dc),      0,      0),
            (0,     -1/T_diesel,    -1/(T_diesel * s_diesel)),
            (0,     1/(2*H),        -D_load/(2*H))
        ), dtype=TORCH_DTYPE, device=device)

        self.B = torch.tensor((
            (omega_b * alpha_ce / C_dc, -omega_b * beta_de / C_dc),
            (0,     0),
            ( (v_sce - 2*R_sc*i_se)/(2*H),  0)
        ), dtype=TORCH_DTYPE, device=device)

        self.G = torch.tensor((
                (0, ),
                (0, ),
                (-1/(2*H), )
        ), dtype=TORCH_DTYPE, device=device)

        # Capture some (arbitrary) dependence between voltage/freq variation and disturb
        #    Note: Doesn't always solve, but solves for random seed 10
        # self.C = torch.tensor((-0.05, 0, 0.05), dtype=TORCH_DTYPE, device=device).unsqueeze(0)
        self.C = 5.0 * torch.tensor(np.random.randn(3), dtype=TORCH_DTYPE, device=device).unsqueeze(0)
        self.D = torch.zeros(self.wq, self.m, dtype=TORCH_DTYPE, device=device)

        # Objective: Assign weight 1 to entries in output vector y, 
        #    and small weight to other values
        self.Q = torch.tensor((
            (1,  0,    0),
            (0,  0.1,  0),
            (0,  0,    1)
        ), dtype=TORCH_DTYPE, device=device)

        self.R = torch.tensor((
            (0.1,   0),
            (0,     0.1)
        ), dtype=TORCH_DTYPE, device=device)

        self.disturb_f = dm.NLDIDisturbModel(self.C, self.D, self.n, self.m, self.wp)
        if device is not None:
            self.disturb_f.to(device=device, dtype=TORCH_DTYPE)

        self.adversarial_disturb_f = None

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

    def get_nldi_linearization(self):
        return self.A, self.B, self.G, self.C, self.D, self.Q, self.R

    def gen_states(self, num_states, device=None):
        return torch.tensor(np.random.rand(num_states, self.n), device=device, dtype=TORCH_DTYPE)

    def __copy__(self):
        new_env = MicrogridEnv.__new__(MicrogridEnv)
        new_env.__dict__.update(self.__dict__)
        return new_env
