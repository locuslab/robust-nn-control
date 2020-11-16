
def get_args():
    args = type('', (), {})()

    args.num_env_steps = 1e7
    args.num_processes = 8
    args.gamma = 0.99

    args.clip_param = 0.1
    args.ppo_epoch = 4
    args.num_mini_batch = 4
    args.value_loss_coef = 0.5
    args.entropy_coef = 0.01
    args.lr = 2.5e-4
    args.rms_prop_eps = 1e-5
    args.max_grad_norm = 0.5

    args.use_linear_lr_decay = True
    args.use_gae = True
    args.gae_lambda = 0.95
    args.use_proper_time_limits = False
    args.save_interval = 100
    args.log_interval = 10
    args.eval_interval = 100

    return args
