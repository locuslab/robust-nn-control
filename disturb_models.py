import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import norm

from constants import *


class NLDIDisturbModel(nn.Module):
    def __init__(self, C, D, n, m, wp):
        super().__init__()
        self.C = C
        self.D = D
        self.net = nn.Sequential(nn.Linear(n + m, 50), nn.Sigmoid(),
                                 nn.Linear(50, 50), nn.Sigmoid())
        self.disturb_layer = nn.Linear(50, wp)
        self.magnitude_layer = nn.Sequential(nn.Linear(50, 1), nn.Tanh())
        list(self.magnitude_layer.parameters())[-1].data *= 10

        self.disturb_size = wp
        self.disturbance = None
    
    def forward(self, x, u, t):
        if self.disturbance is None:
            y = self.net(torch.cat((x, u), dim=1))
            disturb = self.disturb_layer(y)
            magnitude = self.magnitude_layer(y)
        else:
            disturb = self.disturbance
            magnitude = 1
        disturb_norm = torch.norm(disturb, dim=1)
        max_norm = torch.norm(x @ self.C.T + u @ self.D.T, dim=1)
        p = (disturb / disturb_norm.unsqueeze(1)) * max_norm.unsqueeze(1) * magnitude
        return p


class MultiNLDIDisturbModel(nn.Module):
    def __init__(self, bs, C, D, n, m, wp):
        super().__init__()
        self.C = C
        self.D = D
        self.bs = bs
        self.net = nn.Sequential(nn.Linear(self.bs * (n + m), 50), nn.Sigmoid(),
                                 nn.Linear(50, 50), nn.Sigmoid(),
                                 nn.Linear(50, self.bs * wp))
    
    def forward(self, x, u, t):
        disturb = self.net(torch.cat((x, u), dim=1).reshape([1, -1])).reshape([self.bs, -1])
        disturb_norm = torch.norm(disturb, dim=1)
        max_norm = torch.norm(x @ self.C.T + u @ self.D.T, dim=1)
        p = (disturb / disturb_norm.unsqueeze(1)) * max_norm.unsqueeze(1)
        return p
    
    def reset(self):
        def weight_reset(m):
            if isinstance(m, nn.Linear):
                m.reset_parameters()
        
        self.net.apply(weight_reset)


class PLDIDisturbModel(nn.Module):
    def __init__(self, n, m, L):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n + m, 50), nn.ReLU(),
                                 nn.Linear(50, 50), nn.ReLU(),
                                 nn.Linear(50, L), nn.Softmax(1))

        self.disturb_size = L
        self.disturbance = None

    def forward(self, x, u, t):
        if self.disturbance is None:
            disturb = self.net(torch.cat((x, u), dim=1))
        else:
            disturb = nn.Softmax(1)(self.disturbance)
        return disturb


class MultiPLDIDisturbModel(nn.Module):
    def __init__(self, bs, n, m, L):
        super().__init__()
        self.bs = bs
        self.net = nn.Sequential(nn.Linear(self.bs * (n + m), 50), nn.Sigmoid(),
                                 nn.Linear(50, 50), nn.Sigmoid(),
                                 nn.Linear(50, self.bs * L))
        self.softmax = nn.Softmax(1)

    def forward(self, x, u, t):
        return self.softmax(self.net(torch.cat((x, u), dim=1).reshape([1, -1])).reshape([self.bs, -1]))

    def reset(self):
        def weight_reset(m):
            if isinstance(m, nn.Linear):
                m.reset_parameters()
        self.net.apply(weight_reset)


class HinfDisturbModel(nn.Module):
    def __init__(self, n, m, wp, T):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n + m, 50), nn.Sigmoid(),
                                 nn.Linear(50, 50), nn.Sigmoid())
        self.disturb_layer = nn.Linear(50, wp)
        self.magnitude_layer = nn.Sequential(nn.Linear(50, 1), nn.Tanh())
        list(self.magnitude_layer.parameters())[-1].data *= 10
        self.T = T

        self.disturb_size = wp
        self.disturbance = None

    def forward(self, x, u, t):
        if self.disturbance is None:
            y = self.net(torch.cat((x, u), dim=1))
            disturb = self.disturb_layer(y)
            magnitude = self.magnitude_layer(y)
        else:
            disturb = self.disturbance
            magnitude = 1

        disturb_norm = torch.norm(disturb, dim=1)
        if type(t) == torch.Tensor:
            t = t.detach().cpu().numpy()
        max_norm = torch.tensor(20 * norm.pdf(2 * t/self.T), device=x.device).reshape((-1, 1))
        p = (disturb / disturb_norm.unsqueeze(1)) * max_norm * magnitude
        return p


class MultiHinfDisturbModel(nn.Module):
    def __init__(self, bs, n, m, wp, T):
        super().__init__()
        self.bs = bs
        self.net = nn.Sequential(nn.Linear(self.bs * (n + m), 50), nn.Sigmoid(),
                                 nn.Linear(50, 50), nn.Sigmoid(),
                                 nn.Linear(50, self.bs * wp))
        self.T = T
    
    def forward(self, x, u, t):
        disturb = self.net(torch.cat((x, u), dim=1).reshape([1, -1])).reshape([self.bs, -1])
        disturb_norm = torch.norm(disturb, dim=1)
        if type(t) == torch.Tensor:
            t = t.detach().cpu().numpy()
        max_norm = torch.tensor(20 * norm.pdf(2 * t/self.T), device=x.device).reshape((-1, 1))
        p = (disturb / disturb_norm.unsqueeze(1)) * max_norm
        return p
    
    def reset(self):
        def weight_reset(m):
            if isinstance(m, nn.Linear):
                m.reset_parameters()
        
        self.net.apply(weight_reset)


class MBAdvDisturbModel(nn.Module):
    def __init__(self, env, pi, disturb_model, dt,
                 step_type='euler', lr=0.0025, horizon=100, num_iters=100, change_thresh=0.001, update_freq=100, hinf_loss=False):
        super().__init__()
        self.dt = dt
        self.step_type = step_type
        self.lr = lr
        self.horizon = horizon
        self.num_iters = num_iters
        self.change_thresh = change_thresh
        self.update_freq = update_freq
        self.hinf_loss = hinf_loss
        
        self.env = env.__copy__()
        self.pi = pi

        self.disturb_model = disturb_model
        self.num_steps = 0

    def update(self, x_in):
        if self.num_steps % self.update_freq == 0:
            self.env.adversarial_disturb_f = self.disturb_model

            opt = optim.Adam(self.disturb_model.net.parameters(), lr=self.lr)
            x_in = x_in.detach()

            # print('')
            # print('Optimizing...')
            prev_total_cost = np.inf
            for i in range(self.num_iters):
                opt.zero_grad()

                x = x_in
                total_cost = 0
                disturb_norm = 0
                for t in range(self.horizon):
                    u = self.pi(x)
                    x, cost = self.env.step(x, u, t, self.dt, self.step_type, adversarial=True)
                    total_cost += cost
                    
                    if self.hinf_loss:
                        disturb_norm += torch.norm(self.env.disturb, p=2, dim=1)

                if self.hinf_loss:
                    total_cost = (total_cost / disturb_norm).mean()
                else:
                    total_cost = total_cost.mean()
                if torch.isnan(total_cost) or torch.abs(prev_total_cost - total_cost)/total_cost < self.change_thresh:
                    break
                prev_total_cost = total_cost

                (-total_cost).backward(retain_graph=True)
                opt.step()

        self.num_steps += 1

    def forward(self, x_in, u_in, t):
        return self.disturb_model(x_in, u_in, t)

    def set_policy(self, policy):
        del self.pi
        self.pi = policy
        self.reset()

    def reset(self):
        self.disturb_model.reset()
        self.num_steps = 0
