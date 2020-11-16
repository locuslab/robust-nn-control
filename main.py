import numpy as np
import scipy.linalg as la
import cvxpy as cp
import torch
import torch.optim as optim
import argparse
import setproctitle
import os
from gym import spaces
import tqdm

import policy_models as pm
import disturb_models as dm
import robust_mpc as rmpc

from envs.random_nldi_env import RandomNLDIEnv
from envs.cartpole import CartPoleEnv
from envs.quadrotor_env import QuadrotorEnv
from envs.random_pldi_env import RandomPLDIEnv
from envs.random_hinf_env import RandomHinfEnv
from envs.microgrid import MicrogridEnv

from constants import *

from rl.ppo import PPO
from rl.rarl_ppo import RARLPPO
from rl.model import Policy
from rl.storage import RolloutStorage
from rl import trainer
from rl import arguments
from envs.rl_wrapper import RLWrapper

# import ipdb
# import sys
# from IPython.core import ultratb
# sys.excepthook = ultratb.FormattedTB(mode='Verbose',
#      color_scheme='Linux', call_pdb=1)


def main():
    parser = argparse.ArgumentParser(
        description='Run robust control experiments.')
    parser.add_argument('--baseLR', type=float, default=1e-3,
                        help='learning rate for non-projected DPS')
    parser.add_argument('--robustLR', type=float, default=1e-4,
                        help='learning rate for projected DPS')
    parser.add_argument('--alpha', type=float, default=0.001,
                        help='exponential stability coefficient')
    parser.add_argument('--gamma', type=float, default=20,
                        help='bound on L2 gain of disturbance-to-output map (for H_inf control)')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='max epochs')
    parser.add_argument('--test_frequency', type=int, default=20,
                        help='frequency of testing during training')
    parser.add_argument('--T', type=float, default=2,
                        help='time horizon in seconds')
    parser.add_argument('--dt', type=float, default=0.01,
                        help='time increment')
    parser.add_argument('--testSetSz', type=int, default=50,
                        help='size of test set')
    parser.add_argument('--holdSetSz', type=int, default=50,
                        help='size of holdout set')
    parser.add_argument('--trainBatchSz', type=int, default=20,
                        help='batch size for training')
    parser.add_argument('--stepType', type=str,
                        choices=['euler', 'RK4', 'scipy'], default='RK4',
                        help='method for taking steps during training')
    parser.add_argument('--testStepType', type=str,
                        choices=['euler', 'RK4', 'scipy'], default='RK4',
                        help='method for taking steps during testing')
    parser.add_argument('--env', type=str,
                        choices=['random_nldi-d0', 'random_nldi-dnonzero', 'random_pldi_env',
                        'random_hinf_env', 'cartpole', 'quadrotor', 'microgrid'],
                        default='random_nldi-d0',
                        help='environment')
    parser.add_argument('--envRandomSeed', type=int, default=10,
                        help='random seed used to construct the environment')
    parser.add_argument('--save', type=str,
                        help='prefix to add to save path')
    parser.add_argument('--gpu', type=int, default=0,
                        help='prefix to add to save path')
    parser.add_argument('--evaluate', type=str,
                        help='instead of training, evaluate the models from a given directory'
                             ' (remember to use the same random seed)')
    args = parser.parse_args()

    dt = args.dt
    save_sub = '{}+alpha{}+gamma{}+testSz{}+holdSz{}+trainBatch{}+baselr{}+robustlr{}+T{}+stepType{}+testStepType{}+seed{}+dt{}'.format(
        args.env, args.alpha, args.gamma, args.testSetSz, args.holdSetSz,
        args.trainBatchSz, args.baseLR, args.robustLR, args.T,
        args.stepType, args.testStepType, args.envRandomSeed, dt)
    if args.save is not None:
        save = os.path.join('results', '{}+{}'.format(args.save, save_sub))
    else:
        save = os.path.join('results', save_sub)
    trained_model_dir = os.path.join(save, 'trained_models')
    if not os.path.exists(trained_model_dir):
        os.makedirs(trained_model_dir)
    setproctitle.setproctitle(save_sub)
    
    device = torch.device('cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    
    # Setup
    isD0 = (args.env == 'random_nldi-d0') or (args.env == 'quadrotor')  # no u dependence in disturbance bound
    problem_type = 'nldi'
    if 'random_nldi' in args.env:
        env = RandomNLDIEnv(isD0=isD0, random_seed=args.envRandomSeed, device=device)
    elif args.env == 'random_pldi_env':
        env = RandomPLDIEnv(random_seed=args.envRandomSeed, device=device)
        problem_type = 'pldi'
    elif args.env == 'random_hinf_env':
        env = RandomHinfEnv(T=args.T, random_seed=args.envRandomSeed, device=device)
        problem_type = 'hinf'
    elif args.env == 'cartpole':
        env = CartPoleEnv(random_seed=args.envRandomSeed, device=device)
    elif args.env == 'quadrotor':
        env = QuadrotorEnv(random_seed=args.envRandomSeed, device=device)
    elif args.env == 'microgrid':
        env = MicrogridEnv(random_seed=args.envRandomSeed, device=device)
    else:
        raise ValueError('No environment named %s' % args.env)
    evaluate_dir = args.evaluate
    evaluate = evaluate_dir is not None

    # Test and holdout set of states
    torch.manual_seed(17)
    x_test = env.gen_states(num_states=args.testSetSz, device=device)
    x_hold = env.gen_states(num_states=args.holdSetSz, device=device)
    num_episode_steps = int(args.T / dt)

    if problem_type == 'nldi':
        A, B, G, C, D, Q, R = env.get_nldi_linearization()
        state_dim = A.shape[0]
        action_dim = B.shape[1]

        # Get LQR solutions
        Kct, Pct = get_lqr_tensors(A, B, Q, R, args.alpha, device)

        Kr, Sr = get_robust_lqr_sol(*(v.cpu().numpy() for v in (A, B, G, C, D, Q, R)), args.alpha)
        Krt = torch.tensor(Kr, device=device, dtype=TORCH_DTYPE)
        Prt = torch.tensor(np.linalg.inv(Sr), device=device, dtype=TORCH_DTYPE)
        stable_projection = pm.StableNLDIProjection(Prt, A, B, G, C, D, args.alpha, isD0)

        disturb_model = dm.MultiNLDIDisturbModel(x_test.shape[0], C, D, state_dim, action_dim, env.wp)
        disturb_model.to(device=device, dtype=TORCH_DTYPE)

    elif problem_type == 'pldi':
        A, B, Q, R = env.get_pldi_linearization()
        state_dim = A.shape[1]
        action_dim = B.shape[2]

        # Get LQR solutions
        Kct, Pct = get_lqr_tensors(A.mean(0), B.mean(0), Q, R, args.alpha, device)

        Kr, Sr = get_robust_pldi_policy(*(v.cpu().numpy() for v in (A, B, Q, R)), args.alpha)
        Krt = torch.tensor(Kr, device=device, dtype=TORCH_DTYPE)
        Prt = torch.tensor(np.linalg.inv(Sr), device=device, dtype=TORCH_DTYPE)
        stable_projection = pm.StablePLDIProjection(Prt, A, B)

        disturb_model = dm.MultiPLDIDisturbModel(x_test.shape[0], state_dim, action_dim, env.L)
        disturb_model.to(device=device, dtype=TORCH_DTYPE)
    
    elif problem_type == 'hinf':
        A, B, G, Q, R = env.get_hinf_linearization()
        state_dim = A.shape[0]
        action_dim = B.shape[1]

        # Get LQR solutions
        Kct, Pct = get_lqr_tensors(A, B, Q, R, args.alpha, device)

        Kr, Sr, mu = get_robust_hinf_policy(*(v.cpu().numpy() for v in (A, B, G, Q, R)), args.alpha, args.gamma)
        Krt = torch.tensor(Kr, device=device, dtype=TORCH_DTYPE)
        Prt = torch.tensor(np.linalg.inv(Sr), device=device, dtype=TORCH_DTYPE)
        stable_projection = pm.StableHinfProjection(Prt, A, B, G, Q, R, args.alpha, args.gamma, 1/mu)

        disturb_model = dm.MultiHinfDisturbModel(x_test.shape[0], state_dim, action_dim, env.wp, args.T)
        disturb_model.to(device=device, dtype=TORCH_DTYPE)

    else:
        raise ValueError('No problem type named %s' % problem_type)

    adv_disturb_model = dm.MBAdvDisturbModel(env, None, disturb_model, dt, horizon=num_episode_steps//5, update_freq=num_episode_steps//20)
    env.adversarial_disturb_f = adv_disturb_model

    ###########################################################
    # LQR baselines
    ###########################################################

    ### Vanilla LQR (i.e., non-robust, exponentially stable)
    pi_custom_lqr = lambda x: x @ Kct.T
    adv_disturb_model.set_policy(pi_custom_lqr)

    custom_lqr_perf = eval_model(x_test, pi_custom_lqr, env,
                               step_type=args.testStepType, T=args.T, dt=dt)
    write_results(custom_lqr_perf, 'LQR', save)
    custom_lqr_perf = eval_model(x_test, pi_custom_lqr, env,
                                 step_type=args.testStepType, T=args.T, dt=dt, adversarial=True)
    write_results(custom_lqr_perf, 'LQR-adv', save)

    ### Robust LQR
    pi_robust_lqr = lambda x: x @ Krt.T
    adv_disturb_model.set_policy(pi_robust_lqr)

    robust_lqr_perf = eval_model(x_test, pi_robust_lqr, env,
                                 step_type=args.testStepType, T=args.T, dt=dt)
    write_results(robust_lqr_perf, 'Robust LQR', save)
    robust_lqr_perf = eval_model(x_test, pi_robust_lqr, env,
                                 step_type=args.testStepType, T=args.T, dt=dt, adversarial=True)
    write_results(robust_lqr_perf, 'Robust LQR-adv', save)


    ###########################################################
    # Model-based planning methods
    ###########################################################

    ### Non-robust MBP (starting with robust LQR solution)
    pi_mbp = pm.MBPPolicy(Krt, state_dim, action_dim)
    pi_mbp.to(device=device, dtype=TORCH_DTYPE)
    adv_disturb_model.set_policy(pi_mbp)

    if evaluate:
        pi_mbp.load_state_dict(torch.load(os.path.join(evaluate_dir, 'mbp.pt')))
    else:
        pi_mbp_dict, train_losses, hold_losses, test_losses, test_losses_adv, stop_epoch = \
            train(pi_mbp, x_test, x_hold, env,
                  lr=args.baseLR, batch_size=args.trainBatchSz, epochs=args.epochs, T=args.T, dt=dt, step_type=args.stepType,
                  test_frequency=args.test_frequency, save_dir=save, model_name='mbp', device=device)
        save_results(train_losses, hold_losses, test_losses, test_losses_adv, save, 'mbp', pi_mbp_dict, epoch=stop_epoch,
                     is_final=True)
        torch.save(pi_mbp_dict, os.path.join(trained_model_dir, 'mbp.pt'))

    pi_mbp_perf = eval_model(x_test, pi_mbp, env,
                            step_type=args.testStepType, T=args.T, dt=dt)
    write_results(pi_mbp_perf, 'MBP', save)
    pi_mbp_perf = eval_model(x_test, pi_mbp, env,
                            step_type=args.testStepType, T=args.T, dt=dt, adversarial=True)
    write_results(pi_mbp_perf, 'MBP-adv', save)


    ### Robust MBP (starting with robust LQR solution)
    pi_robust_mbp = pm.StablePolicy(pm.MBPPolicy(Krt, state_dim, action_dim), stable_projection)
    pi_robust_mbp.to(device=device, dtype=TORCH_DTYPE)
    adv_disturb_model.set_policy(pi_robust_mbp)

    if evaluate:
        pi_robust_mbp.load_state_dict(torch.load(os.path.join(evaluate_dir, 'robust_mbp.pt')))
    else:
        pi_robust_mbp_dict, train_losses, hold_losses, test_losses, test_losses_adv, stop_epoch = \
            train(pi_robust_mbp, x_test, x_hold, env,
                  lr=args.robustLR, batch_size=args.trainBatchSz, epochs=args.epochs, T=args.T, dt=dt, step_type=args.stepType,
                  test_frequency=args.test_frequency, save_dir=save, model_name='robust_mbp', device=device)
        save_results(train_losses, hold_losses, test_losses, test_losses_adv, save, 'robust_mbp', pi_robust_mbp_dict, epoch=stop_epoch,
                     is_final=True)
        torch.save(pi_robust_mbp_dict, os.path.join(trained_model_dir, 'robust_mbp.pt'))

    pi_robust_mbp_perf = eval_model(x_test, pi_robust_mbp, env,
                                   step_type=args.testStepType, T=args.T, dt=dt)
    write_results(pi_robust_mbp_perf, 'Robust MBP', save)
    pi_robust_mbp_perf = eval_model(x_test, pi_robust_mbp, env,
                                   step_type=args.testStepType, T=args.T, dt=dt, adversarial=True)
    write_results(pi_robust_mbp_perf, 'Robust MBP-adv', save)


    ###########################################################
    # RL methods
    ###########################################################

    if 'random_nldi' in args.env:
        if isD0:
            rmax = 1000
        else:
            rmax = 1000
    elif args.env == 'random_pldi_env':
        rmax = 10
    elif args.env == 'random_hinf_env':
        rmax = 1000
    elif args.env == 'cartpole':
        rmax = 10
    elif args.env == 'quadrotor':
        rmax = 1000
    elif args.env == 'microgrid':
        rmax = 10
    else:
        raise ValueError('No environment named %s' % args.env)

    rl_args = arguments.get_args()
    linear_controller_K = Krt
    linear_controller_P = Prt
    linear_transform = lambda u, x: u + x @ linear_controller_K.T


    ### Vanilla and robust PPO
    base_ppo_perfs = []
    base_ppo_adv_perfs = []
    robust_ppo_perfs = []
    robust_ppo_adv_perfs = []
    for seed in range(1):
        for robust in [False, True]:
            torch.manual_seed(seed)

            if robust:
                # stable_projection = pm.StableNLDIProjection(linear_controller_P, A, B, G, C, D, args.alpha, isD0=isD0)
                action_transform = lambda u, x: stable_projection.project_action(linear_transform(u, x), x)
            else:
                action_transform = linear_transform

            envs = RLWrapper(env, state_dim, action_dim, gamma=rl_args.gamma,
                             dt=dt, rmax=rmax, step_type='RK4', action_transform=action_transform,
                             num_envs=rl_args.num_processes, device=device)
            eval_envs = RLWrapper(env, state_dim, action_dim, gamma=rl_args.gamma,
                                  dt=dt, rmax=rmax, step_type='RK4', action_transform=action_transform,
                                  num_envs=args.testSetSz, device=device)

            actor_critic = Policy(
                envs.observation_space.shape,
                envs.action_space,
                base_kwargs={'recurrent': False})
            actor_critic.to(device=device, dtype=TORCH_DTYPE)
            agent = PPO(
                actor_critic,
                rl_args.clip_param,
                rl_args.ppo_epoch,
                rl_args.num_mini_batch,
                rl_args.value_loss_coef,
                rl_args.entropy_coef,
                lr=rl_args.lr,
                eps=rl_args.rms_prop_eps,
                max_grad_norm=rl_args.max_grad_norm,
                use_linear_lr_decay=rl_args.use_linear_lr_decay)
            rollouts = RolloutStorage(num_episode_steps, rl_args.num_processes,
                                      envs.observation_space.shape, envs.action_space,
                                      actor_critic.recurrent_hidden_state_size)

            ppo_pi = lambda x: action_transform(actor_critic.act(x, None, None, deterministic=True)[1], x)
            adv_disturb_model.set_policy(ppo_pi)

            if evaluate:
                actor_critic.load_state_dict(torch.load(os.path.join(evaluate_dir,
                                                                     'robust_ppo.pt' if robust else 'ppo.pt')))
            else:
                hold_costs, test_costs, adv_test_costs =\
                    trainer.train(agent, envs, rollouts, device, rl_args,
                                  eval_envs=eval_envs, x_hold=x_hold, x_test=x_test, num_episode_steps=num_episode_steps,
                                  save_dir=os.path.join(save, 'robust_ppo' if robust else 'ppo'),
                                  save_extension='%d' % seed)
                save_results(np.zeros_like(hold_costs), hold_costs, test_costs, adv_test_costs, save,
                             'robust_ppo' if robust else 'ppo', actor_critic.state_dict(),
                             epoch=rl_args.num_env_steps, is_final=True)
                torch.save(actor_critic.state_dict(), os.path.join(trained_model_dir,
                                                                   'robust_ppo.pt' if robust else 'ppo.pt'))

            ppo_perf = eval_model(x_test, ppo_pi, env,
                                  step_type=args.testStepType, T=args.T, dt=dt)
            ppo_adv_perf = eval_model(x_test, ppo_pi, env,
                                      step_type=args.testStepType, T=args.T, dt=dt, adversarial=True)

            if robust:
                robust_ppo_perfs.append(ppo_perf.item())
                robust_ppo_adv_perfs.append(ppo_adv_perf.item())
            else:
                base_ppo_perfs.append(ppo_perf.item())
                base_ppo_adv_perfs.append(ppo_adv_perf.item())

    write_results(base_ppo_perfs, 'PPO', save)
    write_results(robust_ppo_perfs, 'Robust PPO', save)
    write_results(base_ppo_adv_perfs, 'PPO-adv', save)
    write_results(robust_ppo_adv_perfs, 'Robust PPO-adv', save)


    # RARL PPO baseline
    adv_ppo_perfs = []
    adv_ppo_adv_perfs = []
    seed = 0
    torch.manual_seed(seed)

    action_transform = linear_transform

    envs = RLWrapper(env, state_dim, action_dim, gamma=rl_args.gamma,
                     dt=dt, rmax=rmax, step_type='RK4', action_transform=action_transform,
                     num_envs=rl_args.num_processes, device=device, rarl=True)
    eval_envs = RLWrapper(env, state_dim, action_dim, gamma=rl_args.gamma,
                          dt=dt, rmax=rmax, step_type='RK4', action_transform=action_transform,
                          num_envs=args.testSetSz, device=device)

    protagornist_ac = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': False})
    protagornist_ac.to(device=device, dtype=TORCH_DTYPE)
    adversary_ac = Policy(
        envs.observation_space.shape,
        envs.disturb_space,
        base_kwargs={'recurrent': False})
    adversary_ac.to(device=device, dtype=TORCH_DTYPE)
    agent = RARLPPO(
        protagornist_ac,
        adversary_ac,
        rl_args.clip_param,
        rl_args.ppo_epoch,
        rl_args.num_mini_batch,
        rl_args.value_loss_coef,
        rl_args.entropy_coef,
        lr=rl_args.lr,
        eps=rl_args.rms_prop_eps,
        max_grad_norm=rl_args.max_grad_norm,
        use_linear_lr_decay=rl_args.use_linear_lr_decay)
    action_space = spaces.Box(low=0, high=1,
                              shape=(envs.action_space.shape[0]+envs.disturb_space.shape[0],), dtype=NUMPY_DTYPE)
    rollouts = RolloutStorage(num_episode_steps, rl_args.num_processes,
                              envs.observation_space.shape, action_space,
                              protagornist_ac.recurrent_hidden_state_size + adversary_ac.recurrent_hidden_state_size,
                              rarl=True)

    ppo_pi = lambda x: action_transform(protagornist_ac.act(x, None, None, deterministic=True)[1], x)
    adv_disturb_model.set_policy(ppo_pi)

    if evaluate:
        agent.load(evaluate_dir)
    else:
        hold_costs, test_costs, adv_test_costs = \
            trainer.train(agent, envs, rollouts, device, rl_args,
                          eval_envs=eval_envs, x_hold=x_hold, x_test=x_test,
                          num_episode_steps=num_episode_steps,
                          save_dir=os.path.join(save, 'rarl_ppo'),
                          save_extension='%d' % seed)
        save_results(np.zeros_like(hold_costs), hold_costs, test_costs, adv_test_costs, save,
                     'rarl_ppo', protagornist_ac.state_dict(),
                     epoch=rl_args.num_env_steps, is_final=True)
        agent.save(trained_model_dir)
    env.disturb_f.disturbance = None

    ppo_perf = eval_model(x_test, ppo_pi, env,
                          step_type=args.testStepType, T=args.T, dt=dt)
    ppo_adv_perf = eval_model(x_test, ppo_pi, env,
                              step_type=args.testStepType, T=args.T, dt=dt, adversarial=True)

    adv_ppo_perfs.append(ppo_perf.item())
    adv_ppo_adv_perfs.append(ppo_adv_perf.item())

    write_results(adv_ppo_perfs, 'RARL PPO', save)
    write_results(adv_ppo_adv_perfs, 'RARL PPO-adv', save)


    ###########################################################
    # MPC baselines
    ###########################################################

    ### Robust MPC (not implemented for H_infinity settings)
    if problem_type != 'hinf':
        if problem_type == 'nldi':
            robust_mpc_model = rmpc.RobustNLDIMPC(A, B, G, C, D, Q, R, Krt, device)
        else:
            robust_mpc_model = rmpc.RobustPLDIMPC(A, B, Q, R, Krt, device)

        pi_robust_mpc = robust_mpc_model.get_action
        adv_disturb_model.set_policy(pi_robust_mpc)

        robust_mpc_perf = eval_model(x_test, pi_robust_mpc, env,
                                step_type=args.testStepType, T=args.T, dt=dt, adversarial=True)
        write_results(robust_mpc_perf, 'Robust MPC-adv', save)



def get_lqr_tensors(At, Bt, Qt, Rt, alpha, device):
    K, S = get_custom_lqr_sol(*(v.cpu().numpy() for v in (At, Bt, Qt, Rt)), alpha)
    Kt = torch.tensor(K, device=device, dtype=TORCH_DTYPE)
    Pt = torch.tensor(np.linalg.inv(S), device=device, dtype=TORCH_DTYPE)

    return Kt, Pt


def get_custom_lqr_sol(A, B, Q, R, alpha):
    n, m = B.shape
    S = cp.Variable((n, n), symmetric=True)
    Y = cp.Variable((m, n))

    R_sqrt = la.sqrtm(R)
    f = cp.trace(S @ Q) + cp.matrix_frac(Y.T @ R_sqrt, S)

    # Exponential stability constraints from LMI book
    cons = [S >> np.eye(n)]  # make LMI non-homogeneous
    cons += [A @ S + S @ A.T + B @ Y + Y.T @ B.T << -alpha * S]

    cp.Problem(cp.Minimize(f), cons).solve()
    K = np.linalg.solve(S.value, Y.value.T).T
    S = S.value

    return np.array(K), np.array(S)


def get_robust_lqr_sol(A, B, G, C, D, Q, R, alpha):
    n, m = B.shape
    wq = C.shape[0]

    S = cp.Variable((n, n), symmetric=True)
    Y = cp.Variable((m, n))
    mu = cp.Variable()

    R_sqrt = la.sqrtm(R)
    f = cp.trace(S @ Q) + cp.matrix_frac(Y.T @ R_sqrt, S)

    cons_mat = cp.bmat((
        (A @ S + S @ A.T + cp.multiply(mu, G @ G.T) + B @ Y + Y.T @ B.T + alpha * S, S @ C.T + Y.T @ D.T),
        (C @ S + D @ Y, -cp.multiply(mu, np.eye(wq)))
    ))
    cons = [S >> 0, mu >= 1e-2] + [cons_mat << 0]

    try:
        prob = cp.Problem(cp.Minimize(f), cons)
        prob.solve(solver=cp.SCS)
    except cp.error.SolverError as e:
        raise ValueError('Solver failed with error: %s \n Try another environment seed' % e)
    K = np.linalg.solve(S.value, Y.value.T).T

    return K, S.value


def get_robust_pldi_policy(A, B, Q, R, alpha):
    L, n, m = B.shape
    S = cp.Variable((n, n), symmetric=True)
    Y = cp.Variable((m, n))

    R_sqrt = la.sqrtm(R)

    f = cp.trace(S @ Q) + cp.matrix_frac(Y.T @ R_sqrt, S)
    cons = [S >> np.eye(n)] + [A[i, :, :] @ S + B[i, :, :] @ Y + S @ A[i, :, :].T + Y.T @ B[i, :, :].T << -alpha * S for i in range(A.shape[0])]
    prob = cp.Problem(cp.Minimize(f), cons)
    prob.solve(solver=cp.MOSEK)
    K = np.linalg.solve(S.value, Y.value.T).T
    return K, S.value


def get_robust_hinf_policy(A, B, G, Q, R, alpha, gamma):
    n, m = B.shape
    wq = G.shape[1]

    S = cp.Variable((n, n), symmetric=True)
    Y = cp.Variable((m, n))
    mu = cp.Variable()

    Q_sqrt = la.sqrtm(Q)
    R_sqrt = la.sqrtm(R)
    f = cp.trace(S @ Q) + cp.matrix_frac(Y.T @ R_sqrt, S)

    cons_mat = cp.bmat([[S @ A.T + A @ S + Y.T @ B.T + B @ Y + alpha * S + (mu / gamma ** 2) * G @ G.T,
                         cp.bmat([[S @ Q_sqrt, Y.T @ R_sqrt]])],
                        [cp.bmat([[Q_sqrt @ S], [R_sqrt @ Y]]), -mu * np.eye(m + n)]])
    cons = [S >> np.eye(n), mu >= 0] + [cons_mat << -1e-3 * np.eye(n+m+n)]

    try:
        prob = cp.Problem(cp.Minimize(f), cons)
        prob.solve(solver=cp.SCS)  #cp.MOSEK)
    except cp.error.SolverError as e:
        raise ValueError('Solver failed with error: %s \n Try another environment seed' % e)
    K = np.linalg.solve(S.value, Y.value.T).T
    
    assert np.all(np.linalg.eigvals(S.value) > 0)
    assert np.all(mu.value > 0)
    assert np.all(np.linalg.eigvals(cons_mat.value) <= 0)
    
    return K, S.value, mu.value


def eval_model(x, pi, env, step_type='euler', T=10, dt=0.05, adversarial=False):
    if adversarial:
        env.adversarial_disturb_f.reset()
    loss = 0
    # maxes = torch.ones(6, dtype=TORCH_DTYPE) * -np.inf
    # mins = torch.ones(6, dtype=TORCH_DTYPE) * np.inf
    for t in tqdm.tqdm(range(int(T / dt)), desc='Evaluating agent%s' % (' adversarial' if adversarial else '')):
        u = pi(x)
        if adversarial:
            env.adversarial_disturb_f.update(x)
        x, cost = env.step(x, u, t, step_type=step_type, dt=dt, adversarial=adversarial)
        loss += cost

        # maxes = torch.max(maxes, torch.max(x, dim=0)[0])
        # mins = torch.min(mins, torch.min(x, dim=0)[0])
    return loss.mean()


def train(model, x_test, x_hold, env, batch_size=20, epochs=1000, test_frequency=10, lr=1e-4, T=1,
          dt=0.05, step_type='euler', save_dir=None, model_name=None, device=None, hinf_loss=False):
    opt = optim.Adam(model.parameters(), lr=lr)
    losses = []
    hold_losses = []
    test_losses = []
    test_losses_adv = []
    num_episode_steps = int(T / dt)

    for i in range(epochs+1):
        opt.zero_grad()
        x = env.gen_states(batch_size, device=device)
        loss = 0
        for t in range(num_episode_steps):
            # train
            model.train()
            u = model(x)
            x, cost = env.step(x, u, t, dt=dt, step_type=step_type)
            loss += cost

        losses.append(loss.mean().item())
        print('Epoch {}. Loss: mean/median {:.3f}/{:.3f}, min/max {:.3f}/{:.3f}'
              .format(i, torch.mean(loss), torch.median(loss), torch.min(loss), torch.max(loss)))

        loss.mean().backward()
        opt.step()

        if i % test_frequency == 0:
            print('Testing...')
            env.adversarial_disturb_f.reset()
            xh = x_hold.detach()
            xt = x_test.detach()
            xta = x_test.detach()
            hold_loss = 0
            test_loss = 0
            test_loss_adv = 0
            hold_disturb_norm = 0
            test_disturb_norm = 0
            test_disturb_norm_adv = 0
            for t in range(num_episode_steps):
                # holdout
                model.eval()
                u_hold = model(xh)
                xh, cost_h = env.step(xh, u_hold, t, dt=dt, step_type=step_type)
                hold_loss += cost_h
                if hinf_loss:
                    hold_disturb_norm += torch.norm(env.disturb, p=2, dim=1)

                # test
                model.eval()
                u_test = model(xt)
                xt, cost_t = env.step(xt, u_test, t, dt=dt, step_type=step_type)
                test_loss += cost_t
                if hinf_loss:
                    test_disturb_norm += torch.norm(env.disturb, p=2, dim=1)

                # test adversarial
                env.adversarial_disturb_f.update(xta)
                model.eval()
                u_test_adv = model(xta)
                xta, cost_ta = env.step(xta, u_test_adv, t, dt=dt, step_type=step_type, adversarial=True)
                test_loss_adv += cost_ta
                if hinf_loss:
                    test_disturb_norm_adv += torch.norm(env.disturb, p=2, dim=1)

            hold_losses.append(hold_loss.mean().item())
            test_losses.append(test_loss.mean().item())
            test_losses_adv.append(test_loss_adv.mean().item())

            print('Hold Loss: mean/median {:.3f}/{:.3f}, min/max {:.3f}/{:.3f}'.format(
                torch.mean(hold_loss), torch.median(hold_loss),
                torch.min(hold_loss), torch.max(hold_loss)))
            print('Test Loss: mean/median {:.3f}/{:.3f}, min/max {:.3f}/{:.3f}'.format(
                torch.mean(test_loss), torch.median(test_loss),
                torch.min(test_loss), torch.max(test_loss)))
            print('Test Loss Adv: mean/median {:.3f}/{:.3f}, min/max {:.3f}/{:.3f}'.format(
                torch.mean(test_loss_adv), torch.median(test_loss_adv),
                torch.min(test_loss_adv), torch.max(test_loss_adv)))
            print('')

        # Save intermediate results
        if i % 100 == 0:
            save_results(np.array(losses), np.array(hold_losses), np.array(test_losses), np.array(test_losses_adv),
                         save_dir, model_name, model.state_dict(), epoch=i)

    return model.state_dict(), losses, hold_losses, test_losses, test_losses_adv, i


def save_results(train_losses, hold_losses, test_losses, test_losses_adv,
                 save_dir, model_name, model_dict, epoch, is_final=False):
    model_save_dir = os.path.join(save_dir, model_name)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    np.save(os.path.join(model_save_dir, 'train_losses.npy'), np.array(train_losses))
    np.save(os.path.join(model_save_dir, 'hold_losses.npy'), np.array(hold_losses))
    np.save(os.path.join(model_save_dir, 'test_losses.npy'), np.array(test_losses))
    np.save(os.path.join(model_save_dir, 'test_losses_adv.npy'), np.array(test_losses_adv))
    torch.save(model_dict, os.path.join(model_save_dir, 'model-{}.pt'.format(epoch)))
    if is_final:
        torch.save(model_dict, os.path.join(model_save_dir, 'model.pt'))


def write_results(test_loss, model_name, save_dir):
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    result_str = '{}: {}\n'.format(model_name, test_loss)
    print(result_str)
    with open(os.path.join(save_dir, 'results.txt'), 'a') as f:
        f.write(result_str)


if __name__ == '__main__':
    main()
