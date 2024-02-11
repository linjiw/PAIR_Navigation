# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from algos import PPO, RolloutStorage, ACAgent

try:

    from models import MultigridNetworkTaskEmbedSingleStepContinuous

except:
    print("Failed loading MiniGrid Networks in util/make_agent.py")



def model_for_multigrid_agent(env):


    adversary_observation_space = env.adversary_observation_space
    adversary_action_space = env.adversary_action_space

    #adversary_random_z_dim = adversary_observation_space['random_z'].shape[0]

    model = MultigridNetworkTaskEmbedSingleStepContinuous(
        observation_space=adversary_observation_space,
        action_space=adversary_action_space,
        scalar_fc=10)


    return model


def model_for_env_agent(env, agent_type='agent'):
    assert agent_type in ['agent', 'adversary_agent', 'adversary_env']
        
    model = model_for_multigrid_agent(env=env)


    return model


def make_agent(name, env, args, device='cpu'):
    # Create model instance
    is_adversary_env = 'env' in name

    use_latent_task = args.use_latent_task if hasattr(args, "use_latent_task") else False
    latent_task_exp = args.latent_task_exp if hasattr(args, "latent_task_exp") else None



    observation_space = env.adversary_observation_space
    action_space = env.adversary_action_space
    entropy_coef = args.adv_entropy_coef
    ppo_epoch = args.adv_ppo_epoch
    num_mini_batch = args.adv_num_mini_batch
    max_grad_norm = args.adv_max_grad_norm
    use_popart = vars(args).get('adv_use_popart', False)


    #CLUTR Specific stuff
    num_steps = 1
    recurrent_arch = False

    recurrent_hidden_size = args.recurrent_hidden_size


    actor_critic = model_for_env_agent(env, name)

    algo = None
    storage = None
    agent = None

    use_proper_time_limits = \
        hasattr(env, 'get_max_episode_steps') \
        and env.get_max_episode_steps() is not None \
        and vars(args).get('handle_timelimits', False)

    # Create PPO
    algo = PPO(
        actor_critic=actor_critic,
        clip_param=args.clip_param,
        ppo_epoch=ppo_epoch,
        num_mini_batch=num_mini_batch,
        value_loss_coef=args.value_loss_coef,
        entropy_coef=entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=max_grad_norm,
        clip_value_loss=args.clip_value_loss,
        log_grad_norm=args.log_grad_norm
    )

    # Create storage
    storage = RolloutStorage(
        model=actor_critic,
        num_steps=num_steps,
        num_processes=args.num_processes,
        observation_space=observation_space,
        action_space=action_space,
        recurrent_hidden_state_size=args.recurrent_hidden_size,
        recurrent_arch=args.recurrent_arch,
        use_proper_time_limits=use_proper_time_limits,
        use_popart=use_popart
    )

    agent = ACAgent(algo=algo, storage=storage).to(device)

    return agent
