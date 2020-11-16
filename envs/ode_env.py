from abc import ABC
import numpy as np
import torch
from scipy import integrate

from constants import *


class ODEEnv(ABC):
    def xdot_f(self, x, u, t):
        raise NotImplementedError

    def xdot_adversarial_f(self, x, u, t):
        raise NotImplementedError

    def cost_f(self, x, u, t):
        raise NotImplementedError

    def gen_states(self, num_states, device=None):
        raise NotImplementedError

    def step(self, x, u, t, dt, step_type, adversarial=False):
        xdot_f = self.xdot_adversarial_f if adversarial else self.xdot_f
        
        if step_type == 'euler':
            x_dot = xdot_f(x, u, t)
            x_next = x + dt * x_dot
            cost = dt * self.cost_f(x, u, t)
        elif step_type == 'RK4':
            # k1 = time_step * xdot_f(x,u)
            # k2 = time_step * xdot_f(x + k1/2, model(x + k1/2))
            # k3 = time_step * xdot_f(x + k2/2, model(x + k2/2))
            # k4 = time_step * xdot_f(x + k3, model(x + k3))
            # x_next = x + (k1 + 2*k2 + 2*k3 + k4)/6
            k1 = xdot_f(x, u, t)
            k2 = xdot_f(x + (dt * k1 / 2), u, t + dt/2)
            k3 = xdot_f(x + (dt * k2 / 2), u, t + dt/2)
            k4 = xdot_f(x + (dt * k3), u, t + dt)
            x_next = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

            cost_k1 = self.cost_f(x, u, t)
            cost_k2 = self.cost_f(x + (dt * k1 / 2), u, t + dt/2)
            cost_k3 = self.cost_f(x + (dt * k2 / 2), u, t + dt/2)
            cost_k4 = self.cost_f(x + (dt * k3), u, t + dt)
            cost = dt * (cost_k1 + 2 * cost_k2 + 2 * cost_k3 + cost_k4) / 6
        elif step_type == 'scipy':
            x_next = torch.zeros_like(x, dtype=TORCH_DTYPE, device=x.device)
            num_x, x_dim = x_next.shape
            cost = torch.zeros([num_x], dtype=TORCH_DTYPE, device=x.device)
            for i in range(num_x):
                u_i = u[i, :].unsqueeze(0)
                f = lambda y, t: torch.cat((xdot_f(torch.Tensor([y[0:x_dim]]), u_i).squeeze(),
                                            self.cost_f(torch.Tensor([y[0:x_dim]]), u_i)), 0).numpy()
                y0 = torch.cat((x[i, :], torch.Tensor([0])), 0).numpy()
                output = integrate.odeint(f, y0, [0, dt])
                x_next[i, :] = torch.Tensor(output[1, :x_dim])
                cost[i] = torch.Tensor(output[1, x_dim:])
        else:
            raise NotImplementedError('Unsupported step type.')

        return x_next, cost


class NLDIEnv(ODEEnv, ABC):
    def get_nldi_linearization(self):
        raise NotImplementedError


class PLDIEnv(ODEEnv, ABC):
    def get_pldi_linearization(self):
        raise NotImplementedError


class HinfEnv(ODEEnv, ABC):
    def get_hinf_linearization(self):
        raise NotImplementedError
