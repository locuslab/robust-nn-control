import torch
import torch.nn as nn
import torch.optim as optim
import os

from rl import utils


class RARLPPO():
    def __init__(self,
                 protagonist_ac,
                 adversarial_ac,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                 use_linear_lr_decay=False):

        self.protagonist_ac = protagonist_ac
        self.adversarial_ac = adversarial_ac

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.lr = lr
        self.use_linear_lr_decay = use_linear_lr_decay
        self.protagonist_optimizer = optim.Adam(protagonist_ac.parameters(), lr=lr, eps=eps)
        self.adversarial_optimizer = optim.Adam(adversarial_ac.parameters(), lr=lr, eps=eps)

        self.num_updates = 0

    def act(self, inputs, rnn_hxs, masks):
        with torch.no_grad():
            _, action, _, _ = self.protagonist_ac.act(inputs, rnn_hxs, masks, deterministic=True)
        return action

    def train_act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, action, action_log_probs, rnn_hxs =\
            self.protagonist_ac.act(inputs, rnn_hxs, masks, deterministic=deterministic)
        adv_value, adv_action, adv_action_log_probs, adv_rnn_hxs =\
            self.adversarial_ac.act(inputs, rnn_hxs, masks, deterministic=deterministic)

        return torch.cat([value, adv_value], dim=-1), \
               torch.cat([action, adv_action], dim=-1), \
               torch.cat([action_log_probs, adv_action_log_probs], dim=-1), \
               None

    def get_value(self, inputs, rnn_hxs, masks):
        value = self.protagonist_ac.get_value(inputs, rnn_hxs, masks)
        adv_value = self.protagonist_ac.get_value(inputs, rnn_hxs, masks)
        return torch.cat([value, adv_value], dim=-1)

    def update(self, rollouts, step, total_steps):
        if self.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(self.protagonist_optimizer, step, total_steps, self.lr)
            utils.update_linear_schedule(self.adversarial_optimizer, step, total_steps, self.lr)

        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean(dim=[0, 1], keepdims=True)) / (
            advantages.std(dim=[0, 1], keepdims=True) + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        switch_freq = 10
        adversarial_update = self.num_updates % (2 * switch_freq) >= switch_freq

        for e in range(self.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, full_actions_batch, \
                   value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample

                actions_batch = full_actions_batch[:, :self.protagonist_ac.num_outputs]
                adv_actions_batch = full_actions_batch[:, self.protagonist_ac.num_outputs:]

                if adversarial_update:
                    values, action_log_probs, dist_entropy, _ = self.adversarial_ac.evaluate_actions(
                        obs_batch, recurrent_hidden_states_batch[:, self.protagonist_ac.num_outputs:], masks_batch,
                        adv_actions_batch)

                    ratio = torch.exp(action_log_probs - old_action_log_probs_batch[:, 1].unsqueeze(-1))
                    value_preds_batch = value_preds_batch[:, 1].unsqueeze(-1)
                    return_batch = return_batch[:, 1].unsqueeze(-1)
                    adv_targ = adv_targ[:, 1].unsqueeze(-1)
                else:
                    values, action_log_probs, dist_entropy, _ = self.protagonist_ac.evaluate_actions(
                        obs_batch, recurrent_hidden_states_batch[:, :self.protagonist_ac.num_outputs], masks_batch,
                        actions_batch)

                    ratio = torch.exp(action_log_probs - old_action_log_probs_batch[:, 0].unsqueeze(-1))
                    value_preds_batch = value_preds_batch[:, 0].unsqueeze(-1)
                    return_batch = return_batch[:, 0].unsqueeze(-1)
                    adv_targ = adv_targ[:, 0].unsqueeze(-1)

                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()
                if adversarial_update:
                    action_loss = -action_loss

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                optimizer = self.adversarial_optimizer if adversarial_update else self.protagonist_optimizer
                params = self.adversarial_ac.parameters() if adversarial_update else self.protagonist_ac.parameters()
                optimizer.zero_grad()
                (value_loss * self.value_loss_coef + action_loss -
                 dist_entropy * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(params, self.max_grad_norm)
                optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        self.num_updates += 1

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch

    def save(self, save_dir):
        torch.save(self.protagonist_ac.state_dict(), os.path.join(save_dir, 'rarl_ppo.pt'))
        torch.save(self.adversarial_ac.state_dict(), os.path.join(save_dir, 'rarl_ppo_adversary.pt'))

    def load(self, save_dir):
        self.protagonist_ac.load_state_dict(torch.load(os.path.join(save_dir, 'rarl_ppo.pt')))
        self.adversarial_ac.load_state_dict(torch.load(os.path.join(save_dir, 'rarl_ppo_adversary.pt')))
