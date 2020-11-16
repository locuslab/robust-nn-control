import numpy as np
import torch

import gym
from gym import spaces

from constants import *


class RLWrapper(gym.Env):

    def __init__(self, env, state_dim, action_dim,
                 rmax=100., gamma=None,
                 dt=0.05, step_type='euler', action_transform=None,
                 num_envs=1, device=None, rarl=False, hinf_loss=False):
        self.env = env
        self.rmax = rmax
        self.dt = dt
        self.step_type = step_type
        self.action_transform = action_transform
        self.num_envs = num_envs
        self.device = device
        self.gamma = gamma
        self.epsilon = 1e-8
        self.cliprew = 10
        self.hinf_loss = hinf_loss

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.max_action_val = 10.
        self.action_space = spaces.Box(low=-self.max_action_val, high=self.max_action_val,
                                       shape=(self.action_dim,), dtype=NUMPY_DTYPE)
        self.max_state_val = 100.
        self.observation_space = spaces.Box(low=-self.max_state_val, high=self.max_state_val,
                                            shape=(self.state_dim,), dtype=NUMPY_DTYPE)
        self.observation_space_low = torch.tensor(self.observation_space.low, device=device)
        self.observation_space_high = torch.tensor(self.observation_space.high, device=device)

        self.rarl = rarl
        self.disturb_space = spaces.Box(low=0, high=1,
                                        shape=(self.env.disturb_f.disturb_size,), dtype=NUMPY_DTYPE)

        self.x = None
        self.episode_reward = None
        self.episode_cost = None
        self.episode_disturb_norm = None
        self.episode_t = None
        self.reset()

    def step(self, u, adversarial=False):
        if adversarial:
            self.env.adversarial_disturb_f.update(self.x)

        if self.rarl:
            disturb = u[:, self.action_dim:]
            u = u[:, :self.action_dim]
            self.env.disturb_f.disturbance = disturb
        else:
            self.env.disturb_f.disturbance = None

        if self.action_transform is not None:
            u = self.action_transform(u, self.x)

        self.x, cost = self.env.step(self.x, u, self.episode_t, self.dt, self.step_type, adversarial=adversarial)
        self.x = self.x.detach()
        cost = cost.detach()
        if self.num_envs == 1:
            self.x.flatten()
        r = torch.clamp(self.rmax - (cost / self.dt), 0, self.rmax) / self.rmax
        self.episode_reward += r
        self.episode_cost += cost
        if self.hinf_loss:
            self.episode_disturb_norm += torch.norm(self.env.disturb, p=2, dim=1)
        self.episode_t += 1
        
        done = torch.max(self.x <= self.observation_space_low,  self.x >= self.observation_space_high)
        if self.num_envs > 1:
            done = done.any(dim=1)
        else:
            done = done.any()

        episode_cost = self.episode_cost / self.episode_disturb_norm if self.hinf_loss else self.episode_cost
        return self.x, r, done, {'episode_reward': self.episode_reward, 'episode_cost': episode_cost}

    def reset(self, x0=None, index=None):
        if x0 is None:
            if self.num_envs == 1:
                self.x = self.env.gen_states(1, device=self.device)[0, :]
            elif index:
                self.x[index, :] = self.env.gen_states(1, device=self.device)[0, :]
            else:
                self.x = self.env.gen_states(self.num_envs, device=self.device)
        else:
            self.x = x0

        if index:
            self.episode_reward[index] = 0
            self.episode_cost[index] = 0
            self.episode_disturb_norm[index] = 0
            self.episode_t[index] = 0
        else:
            self.episode_reward = torch.zeros(self.num_envs, dtype=TORCH_DTYPE, device=self.device)
            self.episode_cost = torch.zeros(self.num_envs, dtype=TORCH_DTYPE, device=self.device)
            self.episode_disturb_norm = torch.zeros(self.num_envs, dtype=TORCH_DTYPE, device=self.device)
            self.episode_t = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        
        return self.x

    def render(self, state_i=0, mode='human'):
        if hasattr(self.env, 'render') and callable(getattr(self.env, 'render')):
            return self.env.render(self.x[state_i, :])
        else:
            raise NotImplementedError

    def close(self):
        if hasattr(self.env, 'close') and callable(getattr(self.env, 'close')):
            return self.env.close()
        else:
            raise NotImplementedError

