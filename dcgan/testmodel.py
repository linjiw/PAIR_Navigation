# import sys
# import os

# # Add the parent directory to sys.path
# parent_directory = os.path.abspath('..')
# sys.path.insert(0, parent_directory)



import matplotlib.pyplot as plt
import argparse
from torchvision.utils import save_image
from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from torch.distributions import Beta
import unittest
import os
# from dcgan import Generator
from dcgan.distributions import Categorical  
# from common import *
# from algo import PPO
# from algo.storage import RolloutStorage
# from algo.agent import ACAgent
# from .popart import PopArt
from dcgan.algos import PPO, RolloutStorage, ACAgent

from dcgan.init_agent import make_agent, train, AdversarialAgentTrainer, make_barn_agent

from dcgan.models import MultigridNetworkTaskEmbedSingleStepContinuous, GridEnvironment, ActorCritic, BARN_ENV, VAE_ENV


def make_agent_env_barn(args):
    obs_size = 10 # Set the size of your observation space
    action_size = 1 # Set the size of your action space
    actor_critic = ActorCritic(obs_size, action_size)
    env = BARN_ENV()
    agent = make_barn_agent(args, env, actor_critic, device='cuda')

    return agent, env


def make_agent_env(args):
    random_z_dim = 50
    adversary_randomz_obs_space = gym.spaces.Box(low=0, high=1.0, shape=(random_z_dim,), dtype=np.float32)
    adversary_observation_space = gym.spaces.Dict({'random_z': adversary_randomz_obs_space})

    latent_dim = args.latent_dim
    action_shape = (latent_dim,)
    latent_dim_min = 0
    latent_dim_max = 1
    adversary_action_space = gym.spaces.Box(low=latent_dim_min, high=latent_dim_max, shape=action_shape, dtype='float32')

    adversary_network = MultigridNetworkTaskEmbedSingleStepContinuous(
        observation_space=adversary_observation_space,
        action_space=adversary_action_space,
        scalar_fc=10
    )
    # adversary_network = MultigridNetworkTaskEmbedSingleStepContinuous(...)  # Initialize with appropriate arguments
    # env = GridEnvironment()
    env = VAE_ENV()
    # final_observation = rollout(adversary_network, env, 10)

    # agent = ACAgent(adversary_network, 10, 10)
    agent = make_agent(args, env, device='cuda')
    return agent, env

# if __init__ == "__main__":
if __name__ == '__main__':
    args = Namespace(
    use_gae=True,
    gamma=0.995,
    gae_lambda=0.95,
    seed=77,
    recurrent_arch='lstm',
    recurrent_agent=True,
    recurrent_adversary_env=True,
    recurrent_hidden_size=256,
    use_global_critic=False,
    lr=0.0001,
    num_steps=10,
    num_processes=1,
    ppo_epoch=5,
    num_mini_batch=1,
    entropy_coef=0.0,
    value_loss_coef=0.5,
    clip_param=0.2,
    clip_value_loss=True,
    adv_entropy_coef=0.0,
    max_grad_norm=0.5,
    algo='ppo',
    ued_algo='paired',
    use_plr=False,
    handle_timelimits=True,
    log_grad_norm=False,
    eps = 1e-5,
    device='cuda',
    train = True,
    n_episodes = 10,
)
    agent, env = make_agent_env(args)
    # n_episodes, max_steps_per_episode = 1, 10
    # train(env, agent, n_episodes, max_steps_per_episode)
    trainer = AdversarialAgentTrainer(args, agent, env)
    trainer.train(args.n_episodes)
