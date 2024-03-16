# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import matplotlib.pyplot as plt
import argparse
from torchvision.utils import save_image
from datetime import datetime
# import wandb
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from torch.distributions import Beta
import unittest
import os
from torchvision.transforms import Resize
from torchvision.transforms.functional import to_pil_image, to_tensor
from torch.distributions import MultivariateNormal
# from jackal_map_create.gen_world_ca import gen_barn_world
import wandb
from dcgan.dcgan import Generator
from dcgan.distributions import Categorical  
from dcgan.common import *
from dcgan.VAE.model import VAE, Encoder, Decoder
from dcgan.VAE.sample import load_model, generate_images, sample_and_save_binary_images
# from algo import PPO
# from algo.storage import RolloutStorage
# from algo.agent import ACAgent
# from .popart import PopArt
from dcgan.algos import PPO, RolloutStorage, ACAgent

# from init_agent import make_agent

# def generate_barn(smooths, fill_pct, folder_name, episode, idx):
    
# def BARN_rollout(network, env, num_steps):
#     observation = env.reset()
#     for step in range(num_steps):
#         with torch.no_grad():
#             # Generate action from the network
#             # _, action, _, _ = network.act({'random_z': observation}, None, None)
#             _, action, _, _ = network.act(observation, None, None)
#             next_observation, reward, done = env.step(action)

#         observation = next_observation
#         if done:
#             observation = env.reset()
#     # Return the final state of the environment
#     return observation

class BARN_ENV:
    def __init__(self) -> None:
        # self.env_para = torch.zeros(1, 2)
        # self.adversary_observation_space = gym.spaces.Dict({'obs': gym.spaces.Box(low=np.array([0.0, -5.0]), high=np.array([1.0, 30.0]), dtype=np.float32)})
        self.adversary_action_space = gym.spaces.Box(low=0.0, high=0.2, shape=(1,), dtype=np.float32)
        self.adversary_observation_space = gym.spaces.Dict({'obs': gym.spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)})
        # self.random_z_dim = 50
        # self.
        # self.fake_observation = torch.zeros(1, 1)
        pass
    def reset(self):
        # self.env_para = torch.zeros(1, 2)
        # self.env_para[0][0] = 4
        # self.env_para[0][1] = 0.1
        # self.performance = torch.zeros(1, 1)
        # self.observation = torch.cat((self.env_para, self.performance), dim=1)
        # self.observation = torch.zeros(1, 2)
        self.observation = torch.randn(1, 10)
        # pass
        # print(f"after reset self.observation: {self.observation}")
        return {'obs': self.observation}
    def step(self, action, episode, idx):
        # action_np = action.detach().numpy()
        # smooths = int(action_np[0][0])
        fill_pct = action.cpu().numpy()
        results_gen = False
        # for i in range(10):
        i = 0
        while i < 1:
            # while not results_gen:
            random_integer = random.randint(1, 100)
            # hash(datetime.now())
            # print(f"fill_pct: {fill_pct}")
            # print(f"random_integer: {random_integer}")
            # if fill_pct > 0.4:
            #     break
            results_gen, metric = gen_barn_world(idx, random_integer, 4, fill_pct, show_metrics=0, folder_name='./../assets/urdf/jackal/')
            if results_gen == True:
                i += 1
                aggregated_metrics = {
                'Distance to Closest Obstacle': [],
                'Average Visibility': [],
                'Dispersion': [],
                'Characteristic Dimension': [],
                'Tortuosity': []
            }
                # print(f"eposide: {idx}, fill_pct {fill_pct}")
                # wandb.log(({"fill_pct": fill_pct}))
                # wandb.log(({"Distance to Closest Obstacle": metric[0]}))
                # wandb.log(({"Average Visibility": metric[1]}))
                # wandb.log(({"Dispersion": metric[2]}))
                # wandb.log(({"Characteristic Dimension": metric[3]}))
                # wandb.log(({"Tortuosity": metric[4]}))

                # print(f"{i} results_gen: {results_gen}")
                # print(f"{i} results_gen: {results_gen}")
            # results_gen = False
        # pass
        # self.fake_observation = copy.deepcopy(self.observation)
        # self.next_obs = torch.zeros(1, 2)
        # self.next_obs[0][0] = action[0][0]
        # obs_low, obs_high = self.adversary_observation_space['obs'].low, self.adversary_observation_space['obs'].high
        # self.next_obs = np.clip(self.next_obs, obs_low, obs_high)
        # print(f"self.next_obs: {self.next_obs}")
        # return {'obs': self.next_obs} , torch.zeros(1, 1), True, {}
        info = {'fill_pct': fill_pct, 'Distance to Closest Obstacle': metric[0], 'Average Visibility': metric[1], 'Dispersion': metric[2], 'Characteristic Dimension': metric[3], 'Tortuosity': metric[4]}

        return self.reset(), torch.zeros(1, 1), True, info
    

class ActorCritic(DeviceAwareModule):
    def __init__(self, obs_size, action_size):
        super(ActorCritic, self).__init__()
        self.recurrent_hidden_state_size = 1
        self.is_recurrent = False
        # Actor
        self.actor_fc = nn.Linear(obs_size, 128)
        self.mu_head = nn.Linear(128, action_size)
        self.log_std_head = nn.Linear(128, action_size)
        # Critic
        self.critic_fc = nn.Linear(obs_size, 128)
        self.value_head = nn.Linear(128, 1)
        self.adversary_observation_space = gym.spaces.Dict({'obs': gym.spaces.Box(low=np.array([0.0, -5.0]), high=np.array([1.0, 30.0]), dtype=np.float32)})
        self.adversary_action_space = gym.spaces.Box(low=0, high=0.2, shape=(1,), dtype=np.float32)
        # self.

    def forward(self, x):
        x = x.to(self.device)  # Ensure input is on the right device
        # Actor
        actor_x = F.relu(self.actor_fc(x))
        mu = self.mu_head(actor_x)
        log_std = self.log_std_head(actor_x)
        std = torch.exp(log_std)
        # Critic
        critic_x = F.relu(self.critic_fc(x))
        state_value = self.value_head(critic_x)
        return mu, std, state_value

    # def act(self, state, ):
    
    def get_value(self, obs, rnn_hxs, masks):
        # core_features, rnn_hxs = self._forward_base(inputs, rnn_hxs, masks)
        state = obs.get('obs')
        state = state.to(self.device)
        mu, std, state_value = self.forward(state)
        return state_value
    
    
    def act(self, obs, rnn_hxs, masks, deterministic=False):
        state = obs.get('obs')
        state = state.to(self.device)
        mu, std, state_value = self.forward(state)
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        action_log_probs = dist.log_prob(action)
        action_low, action_high = self.adversary_action_space.low, self.adversary_action_space.high
        # print(f"action before: {action} action_low {action_low}, action_high {action_high}")
        action = torch.clamp(action, torch.tensor(action_low).to(self.device), torch.tensor(action_high).to(self.device))
        # value = self.evaluate_actions(state, action)
        # print(f"action after: {action}")
        return state_value, action, action_log_probs, rnn_hxs

    def evaluate_actions(self, obs, action, recurrent_hidden_states_batch, masks_batch):
        state = obs.get('obs')
        state, action = state.to(self.device), action.to(self.device)
        mu, std, state_value = self.forward(state)
        # print(f"mu: {mu}")
        # print(f"std: {std}")
        dist = torch.distributions.Normal(mu, std)
        
        action_log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy().mean()
        return state_value, action_log_probs, dist_entropy, mu
    
    



class MultigridNetwork(DeviceAwareModule):
    """
    Actor-Critic module 
    """
    def __init__(self, 
        observation_space, 
        action_space, 
        actor_fc_layers=(32, 32),
        value_fc_layers=(32, 32),
        conv_filters=16,
        conv_kernel_size=3, 
        scalar_fc=5,
        scalar_dim=4,
        random_z_dim=0,
        xy_dim=0,
        recurrent_arch='lstm',
        recurrent_hidden_size=256, 
        random=False):        
        super(MultigridNetwork, self).__init__()

        self.random = random
        self.action_space = action_space
        num_actions = action_space.n

        # Image embeddings
        obs_shape = observation_space['image'].shape
        m = obs_shape[-2] # x input dim
        n = obs_shape[-1] # y input dim
        c = obs_shape[-3] # channel input dim

        self.image_conv = nn.Sequential(
            Conv2d_tf(3, conv_filters, kernel_size=conv_kernel_size, stride=1, padding='valid'),
            nn.Flatten(),
            nn.ReLU()
        )
        self.image_embedding_size = (n-conv_kernel_size+1)*(m-conv_kernel_size+1)*conv_filters
        self.preprocessed_input_size = self.image_embedding_size

        # x, y positional embeddings
        self.xy_embed = None
        self.xy_dim = xy_dim
        if xy_dim:
            self.preprocessed_input_size += 2*xy_dim

        # Scalar embedding
        self.scalar_embed = None
        self.scalar_dim = scalar_dim
        if scalar_dim:
            self.scalar_embed = nn.Linear(scalar_dim, scalar_fc)
            self.preprocessed_input_size += scalar_fc

        self.preprocessed_input_size += random_z_dim
        self.base_output_size = self.preprocessed_input_size

        # RNN
        self.rnn = None
        if recurrent_arch:
            self.rnn = RNN(
                input_size=self.preprocessed_input_size, 
                hidden_size=recurrent_hidden_size,
                arch=recurrent_arch)
            self.base_output_size = recurrent_hidden_size

        # Policy head
        self.actor = nn.Sequential(
            make_fc_layers_with_hidden_sizes(actor_fc_layers, input_size=self.base_output_size),
            Categorical(actor_fc_layers[-1], num_actions)
        )

        # Value head
        self.critic = nn.Sequential(
            make_fc_layers_with_hidden_sizes(value_fc_layers, input_size=self.base_output_size),
            init_(nn.Linear(value_fc_layers[-1], 1))
        )

        apply_init_(self.modules())

        self.train()

    @property
    def is_recurrent(self):
        return self.rnn is not None

    @property
    def recurrent_hidden_state_size(self):
        # """Size of rnn_hx."""
        if self.rnn is not None:
            return self.rnn.recurrent_hidden_state_size
        else:
            return 0

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def _forward_base(self, inputs, rnn_hxs, masks):
        # Unpack input key values
        image = inputs.get('image')

        scalar = inputs.get('direction')
        if scalar is None:
            scalar = inputs.get('time_step')

        x = inputs.get('x')
        y = inputs.get('y')

        in_z = inputs.get('random_z', torch.tensor([], device=self.device))

        in_image = self.image_conv(image)

        if self.xy_embed:
            x = one_hot(self.xy_dim, x, device=self.device)
            y = one_hot(self.xy_dim, y, device=self.device)
            in_x = self.xy_embed(x) 
            in_y = self.xy_embed(y)
        else:
            in_x = torch.tensor([], device=self.device)
            in_y = torch.tensor([], device=self.device)

        if self.scalar_embed:
            in_scalar = one_hot(self.scalar_dim, scalar).to(self.device)
            in_scalar = self.scalar_embed(in_scalar)
        else:
            in_scalar = torch.tensor([], device=self.device)

        in_embedded = torch.cat((in_image, in_x, in_y, in_scalar, in_z), dim=-1)

        if self.rnn is not None:
            core_features, rnn_hxs = self.rnn(in_embedded, rnn_hxs, masks)
        else:
            core_features = in_embedded

        return core_features, rnn_hxs

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        if self.random:
            B = inputs['image'].shape[0]
            action = torch.zeros((B,1), dtype=torch.int64, device=self.device)
            values = torch.zeros((B,1), device=self.device)
            action_log_dist = torch.ones(B, self.action_space.n, device=self.device)
            for b in range(B):
                action[b] = self.action_space.sample()

            return values, action, action_log_dist, rnn_hxs

        core_features, rnn_hxs = self._forward_base(inputs, rnn_hxs, masks)

        dist = self.actor(core_features)
        value = self.critic(core_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_dist = dist.logits

        return value, action, action_log_dist, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        core_features, rnn_hxs = self._forward_base(inputs, rnn_hxs, masks)
        return self.critic(core_features)

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        core_features, rnn_hxs = self._forward_base(inputs, rnn_hxs, masks)

        dist = self.actor(core_features)
        value = self.critic(core_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs
# class MultigridNetworkTaskEmbedSingleStepContinuous(DeviceAwareModule):
#     """
#     Actor-Critic module
#     """
#     def __init__(self, observation_space, action_space, scalar_fc=5, use_popart=False, action_std_init=0.6):
#         super().__init__()

#         self.random_z_dim = observation_space['random_z'].shape[0]
#         self.scalar_fc = scalar_fc
#         self.action_space = action_space
#         self.action_dim = action_space.shape[0]

#         # Embedding and critic layers
#         self.z_embedding = nn.Linear(self.random_z_dim, scalar_fc)
#         self.critic = init_(nn.Linear(self.scalar_fc, 1))
#         self.popart = None
        
#         self.action_var = torch.full((self.action_dim,), action_std_init**2).to(self.device)

#         # Actor layers
#         self.actor_mu = init_relu_(nn.Linear(self.scalar_fc, self.action_dim))
#         self.actor_sigma = nn.Sequential(
#             init_relu_(nn.Linear(self.scalar_fc, self.action_dim)),
#             nn.Softplus()
#         )

#         apply_init_(self.modules())
#         self.train()

#         print(f"Random Z Dimension: {self.random_z_dim}")
#         print(f"Scalar Fully Connected Layer Size: {self.scalar_fc}")
#         print(f"Action Space Shape: {self.action_space.shape}")
#         print(f"Action Dimension: {self.action_dim}")
#         print(f"Base Output Size: {self.scalar_fc}")

#     @property
#     def is_recurrent(self):
#         return False

#     @property
#     def recurrent_hidden_state_size(self):
#         return 1

#     def forward(self, inputs, rnn_hxs, masks):
#         raise NotImplementedError

#     def _forward_base(self, inputs, rnn_hxs, masks):
#         in_z = inputs['random_z']
#         in_z = self.z_embedding(in_z)
#         return in_z

#     # def act(self, inputs, rnn_hxs, masks, deterministic=False):
#     #     in_embedded = self._forward_base(inputs, rnn_hxs, masks)
#     #     value = self.critic(in_embedded)

#     #     # Get mean and standard deviation for the MultivariateNormal distribution
#     #     mu = self.actor_mu(in_embedded)
#     #     sigma = self.actor_sigma(in_embedded)
#     #     print(f"mu.shape: {mu.shape}")
#     #     print(f"sigma.shape: {sigma.shape}")
#     #     # Create the MultivariateNormal distribution and sample an action
#     #     cov_mat = torch.diag(sigma).unsqueeze(dim=0)
#     #     dist = MultivariateNormal(mu, cov_mat)
#     #     action = dist.sample() if not deterministic else mu

#     #     # Calculate the log probability of the action
#     #     action_log_probs = dist.log_prob(action).sum(dim=1).unsqueeze(1)

#     #     return value, action, action_log_probs, rnn_hxs
#     def act(self, inputs, rnn_hxs, masks, deterministic=False):
#         in_embedded = self._forward_base(inputs, rnn_hxs, masks)
#         value = self.critic(in_embedded)

#         mu = self.actor_mu(in_embedded).to(self.device)

#         # Use fixed action variance for the covariance matrix
#         cov_mat = torch.diag(self.action_var).unsqueeze(dim=0).to(self.device)
#         dist = MultivariateNormal(mu, cov_mat)

#         action = dist.sample() if not deterministic else mu
#         action_log_probs = dist.log_prob(action).sum(dim=1).unsqueeze(1)

#         return value, action, action_log_probs, rnn_hxs

#     # def act(self, inputs, rnn_hxs, masks, deterministic=False):
#     #     in_embedded = self._forward_base(inputs, rnn_hxs, masks)
#     #     value = self.critic(in_embedded)

#     #     mu = self.actor_mu(in_embedded)
#     #     sigma = self.actor_sigma(in_embedded)

#     #     # Ensure sigma is a diagonal covariance matrix
#     #     cov_mat = torch.diag(sigma.squeeze())  # Assuming sigma is [1, action_dim]

#     #     dist = MultivariateNormal(mu, covariance_matrix=cov_mat)
#     #     action = dist.sample() if not deterministic else mu

#     #     action_log_probs = dist.log_prob(action).sum(dim=1).unsqueeze(1)

#     #     return value, action, action_log_probs, rnn_hxs

#     def get_value(self, inputs, rnn_hxs, masks):
#         in_embedded = self._forward_base(inputs, rnn_hxs, masks)
#         return self.critic(in_embedded)
#     def evaluate_actions(self, inputs, rnn_hxs, masks, action):
#         in_embedded = self._forward_base(inputs, rnn_hxs, masks)
#         value = self.critic(in_embedded)

#         mu = self.actor_mu(in_embedded)

#         # Use fixed action variance for the covariance matrix
#         cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
#         dist = MultivariateNormal(mu, cov_mat)

#         action_log_probs = dist.log_prob(action).sum(dim=1).unsqueeze(1)
#         dist_entropy = dist.entropy().mean()

#         return value, action_log_probs, dist_entropy, rnn_hxs
#     # def evaluate_actions(self, inputs, rnn_hxs, masks, action):
#     #     in_embedded = self._forward_base(inputs, rnn_hxs, masks)
#     #     value = self.critic(in_embedded)

#     #     mu = self.actor_mu(in_embedded)
#     #     sigma = self.actor_sigma(in_embedded)
#     #     cov_mat = torch.diag(sigma).unsqueeze(dim=0)
#     #     dist = MultivariateNormal(mu, cov_mat)

#     #     action_log_probs = dist.log_prob(action).sum(dim=1).unsqueeze(1)
#     #     dist_entropy = dist.entropy().mean()

#     #     return value, action_log_probs, dist_entropy, rnn_hxs


""""""
class MultigridNetworkTaskEmbedSingleStepContinuous(DeviceAwareModule):
    """
    Actor-Critic module
    """
    def __init__(self,
        observation_space,
        action_space,
        scalar_fc=5,
        use_popart=False):

        super().__init__()

        self.random_z_dim = observation_space['random_z'].shape[0]
        #self.total_time_steps = 1
        #self.time_step_dim = self.total_time_steps + 1

        self.scalar_fc = scalar_fc


        self.random = False
        self.action_space = action_space



        self.action_dim = action_space.shape[0]


        self.z_embedding = nn.Linear(self.random_z_dim, scalar_fc)
        self.base_output_size = self.scalar_fc

        # Value head
        self.critic = init_(nn.Linear(self.base_output_size, 1))

        # Value head
        # if use_popart:
        #     self.critic = init_(PopArt(self.base_output_size, 1))
        #     self.popart = self.critic
        # else:
        self.critic = init_(nn.Linear(self.base_output_size, 1))
        self.popart = None


        self.fc_alpha = nn.Sequential(
            init_relu_(nn.Linear(self.base_output_size, self.action_dim)),
            nn.Softplus()
        )
        self.fc_beta = nn.Sequential(
            init_relu_(nn.Linear(self.base_output_size, self.action_dim)),
            nn.Softplus()
        )

        apply_init_(self.modules())

        self.train()
        
        print(f"Random Z Dimension: {self.random_z_dim}")
        print(f"Scalar Fully Connected Layer Size: {self.scalar_fc}")
        print(f"Action Space Shape: {self.action_space.shape}")
        print(f"Action Dimension: {self.action_dim}")
        print(f"Base Output Size: {self.base_output_size}")

    @property
    def is_recurrent(self):
        return False

    @property
    def recurrent_hidden_state_size(self):
        # """Size of rnn_hx."""
        return 1

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def _forward_base(self, inputs, rnn_hxs, masks):
        # Unpack input key values
        # print(f"inputs: {inputs['random_z']}")
        in_z = inputs['random_z']
        # print(f"in_z: {in_z}")
        in_z = self.z_embedding(in_z)
        # print(f"after forward base in_z: {in_z}")
        return in_z


    def act(self, inputs, rnn_hxs, masks, deterministic=False):

        #core_features, rnn_hxs = self._forward_base(inputs, rnn_hxs, masks)

        in_embedded = self._forward_base(inputs, rnn_hxs, masks)

        value = self.critic(in_embedded)

        # All B x 3
        alpha = 1 + self.fc_alpha(in_embedded)
        beta = 1 + self.fc_beta(in_embedded) 
        # print(f"alpha: {alpha}")
        # print(f"beta: {beta}")
        dist = Beta(alpha, beta)
        # mean = dist.mean
        # std = dist.variance ** 0.5
        # print(f"mean: {mean}")
        # print(f"std: {std}")
        action = dist.sample()
        
        # print(f"dist.log_prob(action): {dist.log_prob(action)}")
        # print(f"action: {action}")
        # fake_action_prob = dist.log_prob(action)
        # print(f"fake_action_prob: {fake_action_prob}")
        # mean = torch.mean(action)
        # std = torch.std(action)

        # # Standardize the tensor
        # new_action = (action - mean) / std
        action_log_probs = dist.log_prob(action).sum(dim=1).unsqueeze(1)
        # print(f"action_log_probs: {action_log_probs}")
        # Hack: Just set action log dist to action log probs, since it's not used.
        action_log_dist = action_log_probs
        # return value, action, fake_action_prob, rnn_hxs
        return value, action, action_log_dist, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        in_embedded = self._forward_base(inputs, rnn_hxs, masks)

        return self.critic(in_embedded)

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        in_embedded = self._forward_base(inputs, rnn_hxs, masks)

        value = self.critic(in_embedded)

        action_in_embed = in_embedded

        alpha = 1 + self.fc_alpha(action_in_embed)
        beta = 1 + self.fc_beta(action_in_embed)

        dist = Beta(alpha, beta)

        action_log_probs = dist.log_prob(action).sum(dim=1).unsqueeze(1)

        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


    def save_checkpoint(self, file_path):
        """
        Save the model checkpoint.
        :param file_path: Path to file where the checkpoint will be saved.
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if hasattr(self, 'optimizer') else None
        }, file_path)
        print(f"Checkpoint saved to {file_path}")

    def load_checkpoint(self, file_path, optimizer=None):
        """
        Load the model checkpoint.
        :param file_path: Path to file from where the checkpoint will be loaded.
        :param optimizer: The optimizer to load the state into, if provided.
        """
        checkpoint = torch.load(file_path)
        self.load_state_dict(checkpoint['model_state_dict'])
        if optimizer and 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded from {file_path}")

def print_image_as_dots(image_array):
    for row in image_array:
        for pixel in row:
            # Print a filled dot for 1, space for 0
            print('●' if pixel == 1 else ' ', end='')
        print()  # Newline after each row
 
def vae_sample(generator, action_vector, folder_name, episode, idx, img_shape):
    pass

def sample(generator, action_vector, folder_name, episode, idx, img_shape):
    cuda = torch.cuda.is_available()
    
    # Ensure action_vector is on the correct device
    action_vector = action_vector.cuda() if cuda else action_vector.cpu()
    mean = torch.mean(action_vector)
    std = torch.std(action_vector)

    # Standardize the tensor
    new_action = (action_vector - mean) / std 

    z = new_action.view(-1, *img_shape)  # Reshape action_vector to match latent_dim
    # print(f"z.shape: {z.shape}")
    gen_imgs = generator(z)
    episode = 'pair'
    sample_folder = f"{folder_name}/worlds{episode}"
    os.makedirs(sample_folder, exist_ok=True)
    # os.makedirs(f"{folder_name}", exist_ok=True)
    resize_transform = Resize((30, 30))

    for i, img in enumerate(gen_imgs.data):
        # Save image after moving it back to CPU
        
        save_image(img.cpu(), f"{sample_folder}/sample_{idx}.png", normalize=True)
        # print(f"Saved image to sampled_images/{test_id}_{episode}/sample_{i}.png")
        # if idx == '0':
        #     images = wandb.Image(img.cpu().numpy(), caption=f"PAIR_ID: {idx}")
        #     wandb.log({f"Generated Map": images})
        #     print(f"wandb logged image")
        resized_img = resize_transform(to_pil_image(img.cpu()))

        # Convert to numpy array
        img_array = np.array(resized_img)

        # Apply threshold (binary conversion)
        threshold = 200
        img_array = np.where(img_array > threshold, 0, 1)

        # Save the numpy array as .npy file
        npy_path = f"{sample_folder}/sample_{idx}_{i}.npy"
        np.save(npy_path, img_array)
        # print(f"Saved image to {npy_path}")
        # print_image_as_dots(img_array)
    return gen_imgs.data
        
        
    
class TestMultigridNetwork(unittest.TestCase):

    def setUp(self):
        random_z_dim = 50
        adversary_randomz_obs_space = gym.spaces.Box(low=0, high=1.0, shape=(random_z_dim,), dtype=np.float32)
        adversary_observation_space = gym.spaces.Dict({'random_z': adversary_randomz_obs_space})

        latent_dim = 100
        action_shape = (latent_dim,)
        latent_dim_min = -3
        latent_dim_max = 3
        adversary_action_space = gym.spaces.Box(low=latent_dim_min, high=latent_dim_max, shape=action_shape, dtype='float32')

        self.model = MultigridNetworkTaskEmbedSingleStepContinuous(
            observation_space=adversary_observation_space,
            action_space=adversary_action_space,
            scalar_fc=10
        )
        self.random_z_vectors = []
        self.action_vectors = []
        self.generator = Generator(opt.encoder_dim, opt.latent_dim, opt.img_size, opt.channels)
        self.generator.load_state_dict(torch.load(f"./dcgan/saved_models/generator_epoch_{opt.model_epoch}.pt"))
        self.generator.eval()
        if torch.cuda.is_available():
            self.generator.cuda()
        
        print(f"Setup complete")
    
    def store_and_plot_data(self, random_z, action):
        self.random_z_vectors.append(random_z.numpy())
        self.action_vectors.append(action.detach().numpy())


    def test_act_method(self):
        # Creating mock inputs
        inputs = {'random_z': torch.randn(1, 50)}
        rnn_hxs = torch.zeros(1, 1)
        masks = torch.ones(1, 1)

        # Testing act method
        value, action, action_log_dist, _ = self.model.act(inputs, rnn_hxs, masks)
        self.assertIsNotNone(value)
        self.assertIsNotNone(action)
        self.assertIsNotNone(action_log_dist)
        self.store_and_plot_data(inputs['random_z'], action)
        self.plot_data(inputs['random_z'], action, test_id=1)

        action = action.cuda() if torch.cuda.is_available() else action
        sample(self.generator, action, test_id=1, img_shape=(opt.latent_dim,))

    def test_get_value_method(self):
        # Creating mock inputs
        inputs = {'random_z': torch.randn(1, 50)}
        rnn_hxs = torch.zeros(1, 1)
        masks = torch.ones(1, 1)

        # Testing get_value method
        value = self.model.get_value(inputs, rnn_hxs, masks)
        self.assertIsNotNone(value)

    def test_evaluate_actions_method(self):
        # Creating mock inputs
        inputs = {'random_z': torch.randn(1, 50)}
        rnn_hxs = torch.zeros(1, 1)
        masks = torch.ones(1, 1)
        action = torch.rand(1, 100)

        # Testing evaluate_actions method
        value, action_log_probs, dist_entropy, _ = self.model.evaluate_actions(inputs, rnn_hxs, masks, action)
        self.assertIsNotNone(value)
        self.assertIsNotNone(action_log_probs)
        self.assertIsNotNone(dist_entropy)
        self.store_and_plot_data(inputs['random_z'], action)
        self.plot_data(inputs['random_z'], action, test_id=3)
        # sample(self.generator, action.cpu(), test_id=3, img_shape=(opt.latent_dim,))
        action = action.cuda() if torch.cuda.is_available() else action
        sample(self.generator, action, test_id=3, img_shape=(opt.latent_dim,))

    def plot_data(self, random_z, action, test_id, output_dir='plots'):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.scatter(range(len(random_z[0])), random_z[0])
        plt.title(f'Test {test_id} - Random Z')

        plt.subplot(1, 2, 2)
        plt.scatter(range(len(action[0])), action[0])
        plt.title(f'Test {test_id} - Action')

        plt.savefig(os.path.join(output_dir, f'test_{test_id}_plot.png'))
        plt.close()


class VAE_ENV:
    def __init__(self):
        print("VAE_ENV running...")
        # Initialize Generator
        encoder_dim = 50
        # latent_dim = 100
        self.latent_dim = 20

        self.img_shape = (self.latent_dim,)
        img_size = 32
        channels = 1
        # model_epoch = 1920
        model_epoch = 9971
        # self.generator = Generator(encoder_dim, latent_dim, img_size, channels)
        # self.generator.load_state_dict(torch.load(f"./../dcgan/saved_models/official_generator_epoch_{model_epoch}.pt"))
        # self.generator.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = '/home/linjiw/Downloads/PAIR_Navigation/dcgan/VAE/vae_model_new.pth'
        self.vae = load_model(model_path, self.latent_dim, self.device)
        random_z_dim = 50
        adversary_randomz_obs_space = gym.spaces.Box(low=0, high=1.0, shape=(random_z_dim,), dtype=np.float32)
        self.adversary_observation_space = gym.spaces.Dict({'random_z': adversary_randomz_obs_space})
        # latent_dim = 100
        action_shape = (self.latent_dim,)
        latent_dim_min = -3
        latent_dim_max = 3
        current_time = datetime.now()
        self.sample_img = None
        self.time_string = current_time.strftime("%m_%d_%Y_%H_%M_%S")
        self.adversary_action_space = gym.spaces.Box(low=latent_dim_min, high=latent_dim_max, shape=action_shape, dtype='float32')
        self.global_sample_count = 0
        self.folder_name = './../../../../assets/urdf/jackal'
        # self.folder_name = './../assets/urdf/jackal'
        # if torch.cuda.is_available():
        #     self.generator.cuda()

    # def reset(self):
    #     # Generate random noise as observation
    #     random_noise = torch.randn(1, 50)
    #     return random_noise
    def reset(self):
        # Generate random noise as observation
        random_noise = torch.randn(1, 50).to('cuda')
        # Wrap the observation as a dictionary
        obs_dict = {'random_z': random_noise}
        return obs_dict
    
    def step(self, action, episode, step_id):
        # Process the action with the generator

        # Format the time string as "month_day_year_hour_min_sec"
        # step_id = 'pair'
        random_latent_vectors = action
        _, occupancy_rate, good_map = sample_and_save_binary_images(self.vae, episode, random_latent_vectors, self.device, step_id,save_dir=self.folder_name)
        # self.sample_img = sample(self.generator, action, self.folder_name, episode, str(step_id), self.img_shape)
        self.global_sample_count += 1
        if good_map:
            
            reward = torch.randn(1)
        else:
            reward = torch.zeros(1)
        done = True
        info = {}
        info['occupancy_rate'] = occupancy_rate
        info['good_map'] = good_map
        # print(f"globe_sample_count: {self.global_sample_count}")
        return self.reset(), reward, done, info
    
 

class ClutrEnv:
    def __init__(self):
        # Initialize Generator
        encoder_dim = 50
        latent_dim = 100
        self.img_shape = (latent_dim,)
        img_size = 32
        channels = 1
        # model_epoch = 1920
        # change to VAE and load
        model_epoch = 9971
        self.generator = Generator(encoder_dim, latent_dim, img_size, channels)
        self.generator.load_state_dict(torch.load(f"./../dcgan/saved_models/official_generator_epoch_{model_epoch}.pt"))
        self.generator.eval()
        #-------------------------------
        random_z_dim = 50
        adversary_randomz_obs_space = gym.spaces.Box(low=0, high=1.0, shape=(random_z_dim,), dtype=np.float32)
        self.adversary_observation_space = gym.spaces.Dict({'random_z': adversary_randomz_obs_space})
        latent_dim = 100
        action_shape = (latent_dim,)
        latent_dim_min = -3
        latent_dim_max = 3
        current_time = datetime.now()
        self.sample_img = None
        self.time_string = current_time.strftime("%m_%d_%Y_%H_%M_%S")
        self.adversary_action_space = gym.spaces.Box(low=latent_dim_min, high=latent_dim_max, shape=action_shape, dtype='float32')
        self.global_sample_count = 0
        self.folder_name = './../assets/urdf/jackal'
        if torch.cuda.is_available():
            self.generator.cuda()

    def reset(self):
        # Generate random noise as observation
        random_noise = torch.randn(1, 50).to('cuda')
        # Wrap the observation as a dictionary
        obs_dict = {'random_z': random_noise}
        return obs_dict
    
    def step(self, action, episode, step_id):
        # Process the action with the generator

        # Format the time string as "month_day_year_hour_min_sec"
        # step_id = 'pair'
        self.sample_img = sample(self.generator, action, self.folder_name, episode, str(step_id), self.img_shape)
        self.global_sample_count += 1
        reward = torch.randn(1)
        done = True
        info = {}
        # print(f"globe_sample_count: {self.global_sample_count}")
        return self.reset(), reward, done, info
    
     
        
class GridEnvironment:
    def __init__(self):
        # Initialize Generator
        encoder_dim = 50
        latent_dim = 100
        self.img_shape = (latent_dim,)
        img_size = 32
        channels = 1
        # model_epoch = 1920
        model_epoch = 9971
        self.generator = Generator(encoder_dim, latent_dim, img_size, channels)
        self.generator.load_state_dict(torch.load(f"./../dcgan/saved_models/official_generator_epoch_{model_epoch}.pt"))
        self.generator.eval()
        random_z_dim = 50
        adversary_randomz_obs_space = gym.spaces.Box(low=0, high=1.0, shape=(random_z_dim,), dtype=np.float32)
        self.adversary_observation_space = gym.spaces.Dict({'random_z': adversary_randomz_obs_space})
        latent_dim = 100
        action_shape = (latent_dim,)
        latent_dim_min = -3
        latent_dim_max = 3
        current_time = datetime.now()
        self.sample_img = None
        self.time_string = current_time.strftime("%m_%d_%Y_%H_%M_%S")
        self.adversary_action_space = gym.spaces.Box(low=latent_dim_min, high=latent_dim_max, shape=action_shape, dtype='float32')
        self.global_sample_count = 0
        self.folder_name = './../assets/urdf/jackal'
        if torch.cuda.is_available():
            self.generator.cuda()

    # def reset(self):
    #     # Generate random noise as observation
    #     random_noise = torch.randn(1, 50)
    #     return random_noise
    def reset(self):
        # Generate random noise as observation
        random_noise = torch.randn(1, 50).to('cuda')
        # Wrap the observation as a dictionary
        obs_dict = {'random_z': random_noise}
        return obs_dict
    
    def step(self, action, episode, step_id):
        # Process the action with the generator

        # Format the time string as "month_day_year_hour_min_sec"
        # step_id = 'pair'
        self.sample_img = sample(self.generator, action, self.folder_name, episode, str(step_id), self.img_shape)
        self.global_sample_count += 1
        reward = torch.randn(1)
        done = True
        info = {}
        # print(f"globe_sample_count: {self.global_sample_count}")
        return self.reset(), reward, done, info
    
   
def rollout(network, env, num_steps):
    observation = env.reset()
    for step in range(num_steps):
        with torch.no_grad():
            # Generate action from the network
            _, action, _, _ = network.act({'random_z': observation}, None, None)
            next_observation, reward, done = env.step(action)

        observation = next_observation
        if done:
            observation = env.reset()
    # Return the final state of the environment
    return observation

def compute_final_return(agent, env):
    return 0

# def train(env, agent, n_episodes, max_steps_per_episode):
#     """
#     Train the agent in the given environment.

#     :param env: The gym environment.
#     :param agent: The ACAgent instance.
#     :param n_episodes: Number of episodes to train for.
#     :param max_steps_per_episode: Maximum steps in each episode.
#     """
#     for episode in range(n_episodes):
#         obs = env.reset()
#         # agent.storage.reset()
#         print(f"obs: {obs}")
#         agent.storage.copy_obs_to_index(obs,0)

#         recurrent_hidden_states = torch.zeros([1,  agent.algo.actor_critic.recurrent_hidden_state_size])
#         masks = torch.zeros(1, 1)
#         bad_masks = torch.zeros(1, 1)
#         # level_seeds and cliffhanger_masks are optional and depend on your specific use case

#         for step in range(max_steps_per_episode):
#             # Sample actions
#             with torch.no_grad():
#                 value, action, action_log_dist, recurrent_hidden_states = agent.act(
#                     obs, recurrent_hidden_states, masks)
#                 # action_log_prob = action_log_dist.gather(-1, action)  # Assuming discrete actions
#                 action_log_prob = action_log_dist
#             # Observe reward and next obs
#             # next_obs, reward, done, infos = env.step(action.cpu().numpy())
#             next_obs, reward, done, infos = env.step(action)
#             # Handle any additional flags or data from infos
#             # For example, handling cliffhangers or level replays

#             # Convert done to masks and bad_masks
#             masks.fill_(0.0 if done else 1.0)
#             bad_masks.fill_(0.0 if 'truncated' in infos else 1.0)
#             # level_seeds and cliffhanger_masks logic goes here

#             # Insert the results into the RolloutStorage
#             agent.insert(obs, recurrent_hidden_states, action, action_log_prob, action_log_dist,
#                          value, reward, masks, bad_masks)

#             obs = next_obs

#             if done:
#                 break

#         # After collecting rollouts, compute returns
#         with torch.no_grad():
#             next_value = agent.get_value(next_obs, recurrent_hidden_states, masks)
#         agent.storage.replace_final_return(torch.tensor([0]))

#         agent.storage.compute_returns(next_value,
#                         arguse_gae,
#                         gamma,
#                         gae_lambda):

#         # Perform PPO update
#         agent.update()

#         # Logging and any additional updates

#     print("Training completed.")

# Assuming env, agent, and other parameters are already defined
# train(env, agent, n_episodes, max_steps_per_episode)

# Assuming env, agent and other parameters are already defined
# train(env, agent, n_episodes, max_steps_per_episode)


if __name__ == '__main__':
    pass
    # random_z_dim = 50
    # adversary_randomz_obs_space = gym.spaces.Box(
    # low=0, high=1.0, shape=(random_z_dim,), dtype=np.float32)
    # adversary_observation_space = gym.spaces.Dict(
    #     {'random_z': adversary_randomz_obs_space})
    
    # latent_dim = 100
    # action_shape = (latent_dim,)
    # latent_dim_min = -3
    # latent_dim_max = 3
    # adversary_action_space = gym.spaces.Box(
    # low=latent_dim_min, high=latent_dim_max, shape=action_shape, dtype='float32')

    
    # model = MultigridNetworkTaskEmbedSingleStepContinuous(
    # observation_space=adversary_observation_space,
    # action_space=adversary_action_space,
    # scalar_fc=10)
    # unittest.main()
    # suite = unittest.TestLoader().loadTestsFromTestCase(TestMultigridNetwork)
    # test = TestMultigridNetwork()
    # unittest.TextTestRunner().run(suite)
    # test.plot_data()
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--n_samples", type=int, default=10, help="number of samples to generate")
    # parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    # parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    # parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
    # parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    # parser.add_argument("--model_epoch", type=int, default=1920, help="epoch of saved model to load")
    # parser.add_argument("--encoder_dim", type=int, default=50, help="size of each image dimension")

    # opt = parser.parse_args()
    
    # unittest.main()
    # grid = GridEnvironment()
    # random_z_dim = 50
    # adversary_randomz_obs_space = gym.spaces.Box(low=0, high=1.0, shape=(random_z_dim,), dtype=np.float32)
    # adversary_observation_space = gym.spaces.Dict({'random_z': adversary_randomz_obs_space})

    # latent_dim = 100
    # action_shape = (latent_dim,)
    # latent_dim_min = -3
    # latent_dim_max = 3
    # adversary_action_space = gym.spaces.Box(low=latent_dim_min, high=latent_dim_max, shape=action_shape, dtype='float32')

    # adversary_network = MultigridNetworkTaskEmbedSingleStepContinuous(
    #     observation_space=adversary_observation_space,
    #     action_space=adversary_action_space,
    #     scalar_fc=10
    # )
    # # adversary_network = MultigridNetworkTaskEmbedSingleStepContinuous(...)  # Initialize with appropriate arguments
    # env = GridEnvironment()
    # final_observation = rollout(adversary_network, env, 10)
    
    # # agent = ACAgent(adversary_network, 10, 10)
    # agent = make_agent("adversary_env", env, device='gpu')
    # train(env, adversary_network, 10, 10)
    
    
    
