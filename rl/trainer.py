import os
import time
from collections import deque

from rl import utils
from constants import *


def train(agent, envs, rollouts, device, args, save_dir,
          eval_envs=None, x_hold=None, x_test=None, save_extension=None,
          num_episode_steps=100):
    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=100)
    
    hold_costs = []
    test_costs = []
    adv_test_costs = []

    min_cost = np.inf
    if save_extension is not None:
        save_dir = os.path.join(save_dir, save_extension)
    try:
        os.makedirs(save_dir)
    except OSError:
        pass
    agent.save(save_dir)

    start = time.time()
    num_updates = int(args.num_env_steps) // num_episode_steps // args.num_processes
    for j in range(num_updates):
        obs = envs.reset()

        masks = torch.zeros((envs.num_envs, 1), dtype=TORCH_DTYPE, device=obs.device)
        bad_masks = torch.ones((envs.num_envs, 1), dtype=TORCH_DTYPE, device=obs.device)
        recurrent_hidden_states = torch.ones((envs.num_envs, 1), dtype=TORCH_DTYPE, device=obs.device)
        rollouts.reset(obs, recurrent_hidden_states, masks, bad_masks)

        for step in range(num_episode_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = agent.train_act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for i, d in enumerate(done):
                if d:
                    episode_rewards.append(infos['episode_reward'][i].cpu().numpy())
                    obs = envs.reset(index=i)

            # If done then clean the history of observations.
            masks = torch.tensor([[0.0] if done_ else [1.0] for done_ in done], dtype=TORCH_DTYPE)
            bad_masks = torch.ones((envs.num_envs, 1), dtype=TORCH_DTYPE)
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward.unsqueeze(1), masks, bad_masks)

        with torch.no_grad():
            next_value = agent.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts, j, num_updates)

        rollouts.after_update()

        for i in range(envs.num_envs):
            episode_rewards.append(infos['episode_reward'][i].cpu().numpy())

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * num_episode_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                    .format(j, total_num_steps,
                            int(total_num_steps / (end - start)),
                            len(episode_rewards), np.mean(episode_rewards),
                            np.median(episode_rewards), np.min(episode_rewards),
                            np.max(episode_rewards), dist_entropy, value_loss,
                            action_loss))

        if eval_envs is not None \
                and (args.eval_interval is not None and j % args.eval_interval == 0):
            hold_cost = evaluate(agent, eval_envs, device, num_episode_steps, x_test=x_hold, set_name='x_hold')
            hold_costs.append(hold_cost.cpu().numpy())

            test_cost = evaluate(agent, eval_envs, device, num_episode_steps, x_test=x_test, set_name='x_test')
            test_costs.append(test_cost.cpu().numpy())

            eval_envs.env.adversarial_disturb_f.reset()
            adv_test_cost = evaluate(agent, eval_envs, device, num_episode_steps, x_test=x_test, set_name='adv_x_test', adversarial=True)
            adv_test_costs.append(adv_test_cost.detach().cpu().numpy())
            
            if hold_cost < min_cost:
                min_cost = hold_cost
                agent.save(save_dir)

    agent.load(save_dir)
    return hold_costs, test_costs, adv_test_costs


def evaluate(agent, eval_env, device, max_num_steps, x_test=None, set_name='', adversarial=False):
    if x_test is not None:
        obs = eval_env.reset(x0=x_test)
    else:
        obs = eval_env.reset()
    
    for step in range(max_num_steps):
        action = agent.act(obs, None, None)

        # Obser reward and next obs
        obs, _, done, info = eval_env.step(action, adversarial=adversarial)

    episode_rewards = info['episode_reward']
    episode_costs = info['episode_cost']

    print(" Evaluating {} using {} episodes:  mean/median cost {:.3f}/{:.3f}, min/max cost {:.3f}/{:.3f}   mean/median reward {:.3f}/{:.3f}, min/max reward {:.3f}/{:.3f}\n".format(
        set_name, episode_costs.shape[0],
        torch.mean(episode_costs), torch.median(episode_costs),
        torch.min(episode_costs), torch.max(episode_costs),
        torch.mean(episode_rewards), torch.median(episode_rewards),
        torch.min(episode_rewards), torch.max(episode_rewards)))
    
    return torch.mean(episode_costs)
