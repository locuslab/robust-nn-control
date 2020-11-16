import numpy as np
import torch
import os

from envs import ode_env
import disturb_models as dm
from constants import *


class CartPoleEnv(ode_env.NLDIEnv):

    def __init__(self, l=1, m_cart=1, m_pole=1, g=9.81, Q=None, R=None, random_seed=None, device=None):
        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed+1)

        self.l = l
        self.m_cart = m_cart
        self.m_pole = m_pole
        self.g = g

        self.n, self.m, = 4, 1

        # TODO: have reasonable objective?
        self.Q = Q
        self.R = R
        if Q is None:
            Q = np.random.randn(self.n, self.n)
            Q = Q.T @ Q
            # Q = np.eye(self.n)
            self.Q = torch.tensor(Q, dtype=TORCH_DTYPE, device=device)
        if R is None:
            R = np.random.randn(self.m, self.m)
            R = R.T @ R
            # R = np.eye(self.m)
            self.R = torch.tensor(R, dtype=TORCH_DTYPE, device=device)

        # TODO: hacky, assumes call from main.py in top level directory
        array_path = os.path.join('problem_gen', 'cartpole')
        self.A = torch.tensor(np.load(os.path.join(array_path, 'A.npy')), dtype=TORCH_DTYPE, device=device)
        self.B = torch.tensor(np.load(os.path.join(array_path, 'B.npy')), dtype=TORCH_DTYPE, device=device)
        self.G_lin = torch.tensor(np.load(os.path.join(array_path, 'G.npy')), dtype=TORCH_DTYPE, device=device)
        self.C_lin = torch.tensor(np.load(os.path.join(array_path, 'C.npy')), dtype=TORCH_DTYPE, device=device)
        self.D_lin = torch.tensor(np.load(os.path.join(array_path, 'D.npy')), dtype=TORCH_DTYPE, device=device)

        disturb_n = 2
        self.G_disturb = torch.tensor(np.random.randn(self.n, disturb_n), dtype=TORCH_DTYPE, device=device)
        self.C_disturb = torch.tensor(0.1 * np.random.randn(disturb_n, self.n), dtype=TORCH_DTYPE, device=device)
        self.D_disturb = torch.tensor(0.001 * np.random.randn(disturb_n, self.m), dtype=TORCH_DTYPE, device=device)

        self.G = torch.cat([self.G_lin, self.G_disturb], dim=1)
        self.C = torch.cat([self.C_lin, self.C_disturb], dim=0)
        self.D = torch.cat([self.D_lin, self.D_disturb], dim=0)

        self.wp, self.wq = self.G.shape[1], self.C.shape[0]

        self.disturb_f = dm.NLDIDisturbModel(self.C_disturb, self.D_disturb, self.n, self.m, self.G_disturb.shape[1])
        if device is not None:
            self.disturb_f.to(device=device, dtype=TORCH_DTYPE)

        self.adversarial_disturb_f = None

        # Max and min values for state and action: [x, xdot, theta, thetadot, u]
        self.yumax = torch.tensor([1.2, 1.0, 0.1, 1.0, 10], dtype=TORCH_DTYPE, device=device)
        self.yumin = torch.tensor([-1.2, -1.0, -0.1, -1.0, -10], dtype=TORCH_DTYPE, device=device)

        self.y_0_max = torch.tensor([1.0, 0.0, 0.1, 0.0], dtype=TORCH_DTYPE, device=device)
        self.y_0_min = torch.tensor([-1.0, -0.0, -0.1, -0.0], dtype=TORCH_DTYPE, device=device)

        self.viewer = None

    # Keeping external interface, but renaming internally
    def xdot_f(self, state, u_in, t):
        # x = state[:, 0]
        x_dot = state[:, 1]
        theta = state[:, 2]
        theta_dot = state[:, 3]

        # limit action magnitude
        if self.m == 1:
            u = torch.clamp(u_in, self.yumin[-1], self.yumax[-1]).squeeze(1)
        else:
            raise NotImplementedError()

        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        temp = 1/(self.m_cart + self.m_pole * (sin_theta * sin_theta))
        x_ddot = temp * (u + self.m_pole * sin_theta * (self.l * (theta_dot**2)
                                                            - self.g * cos_theta))
        theta_ddot = -(1/self.l) * temp * (u * cos_theta
                                          + self.m_pole * self.l * (theta_dot**2) * cos_theta * sin_theta
                                          - (self.m_cart + self.m_pole) * self.g * sin_theta)

        return torch.stack([x_dot, x_ddot, theta_dot, theta_ddot]).T

    def xdot_adversarial_f(self, x, u, t):
        if self.adversarial_disturb_f is None:
            raise ValueError('You must initialize adversarial_disturb_f before running in adversarial mode')

        # # limit action magnitude
        # if self.m == 1:
        #     u = torch.clamp(u_in, self.yumin[-1], self.yumax[-1]).squeeze(1)
        # else:
        #     raise NotImplementedError()

        p = self.adversarial_disturb_f(x, u, t)
        return x @ self.A.T + u @ self.B.T + p @ self.G.T

    def cost_f(self, x, u, t):
        return ((x @ self.Q) * x).sum(-1) + ((u @ self.R) * u).sum(-1)

    def get_nldi_linearization(self):
        return self.A, self.B, self.G, self.C, self.D, self.Q, self.R

    def gen_states(self, num_states, device=None):
        prop = torch.tensor(np.random.rand(num_states, self.n), device=device, dtype=TORCH_DTYPE)
        return self.y_0_max[:self.n].detach()*prop + self.y_0_min[:self.n].detach()*(1-prop)

    def __copy__(self):
        new_env = CartPoleEnv.__new__(CartPoleEnv)
        new_env.__dict__.update(self.__dict__)
        return new_env

    # Copied from Open AI gym: https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
    def render(self, state, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = 10
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.l)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        cartx = state[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-state[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
