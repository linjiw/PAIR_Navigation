# import time
# import gym
# import numpy as np
# import torch
# import copy
# from rl_games.common import env_configurations
# from rl_games.algos_torch import  model_builder

# class BasePlayer(object):
#     def __init__(self, params, created_env = None):
#         self.config = config = params['config']
#         self.load_networks(params)
#         self.env_name = self.config['env_name']
#         self.env_config = self.config.get('env_config', {})
#         self.env_info = self.config.get('env_info')
#         self.clip_actions = config.get('clip_actions', True)
#         self.seed = self.env_config.pop('seed', None)
#         if self.env_info is None:
#             self.env = self.create_env()
#             self.env_info = env_configurations.get_env_info(self.env)
#         else:
#             if created_env is None:
#                 # self.env = self.create_env()
#                 print('[BasePlayer] Using provided env: ', self.env_name)
#                 self.env = created_env
#                 # self.env_info = env_configurations.get_env_info(self.env)
#                 self.env_info = self.env.get_env_info()
#             else:

#                 self.env = config.get('vec_env')
#         self.value_size = self.env_info.get('value_size', 1)
#         self.action_space = self.env_info['action_space']
#         self.num_agents = self.env_info['agents']

#         self.observation_space = self.env_info['observation_space']
#         if isinstance(self.observation_space, gym.spaces.Dict):
#             self.obs_shape = {}
#             for k, v in self.observation_space.spaces.items():
#                 self.obs_shape[k] = v.shape
#         else:
#             self.obs_shape = self.observation_space.shape
#         self.is_tensor_obses = False

#         self.states = None
#         self.player_config = self.config.get('player', {})
#         self.use_cuda = True
#         self.batch_size = 1
#         self.has_batch_dimension = False
#         self.has_central_value = self.config.get('central_value_config') is not None
#         self.device_name = self.config.get('device_name', 'cuda')
#         self.render_env = self.player_config.get('render', False)
#         self.games_num = self.player_config.get('games_num', 500)
#         self.is_determenistic = self.player_config.get('determenistic', True)
#         self.n_game_life = self.player_config.get('n_game_life', 1)
#         self.print_stats = self.player_config.get('print_stats', True)
#         self.render_sleep = self.player_config.get('render_sleep', 0.002)
#         self.max_steps = self.player_config.get('max_steps', 32)
#         self.device = torch.device(self.device_name)

#     def load_networks(self, params):
#         builder = model_builder.ModelBuilder()
#         self.config['network'] = builder.load(params)

#     def _preproc_obs(self, obs_batch):
#         if type(obs_batch) is dict:
#             obs_batch = copy.copy(obs_batch)
#             for k,v in obs_batch.items():
#                 if v.dtype == torch.uint8:
#                     obs_batch[k] = v.float() / 255.0
#                 else:
#                     obs_batch[k] = v
#         else:
#             if obs_batch.dtype == torch.uint8:
#                 obs_batch = obs_batch.float() / 255.0
#         return obs_batch

#     def env_step(self, env, actions):
#         if not self.is_tensor_obses:
#             actions = actions.cpu().numpy()
#         obs, rewards, dones, infos = env.step(actions)
#         if hasattr(obs, 'dtype') and obs.dtype == np.float64:
#             obs = np.float32(obs)
#         if self.value_size > 1:
#             rewards = rewards[0]
#         if self.is_tensor_obses:
#             return self.obs_to_torch(obs), rewards.cpu(), dones.cpu(), infos
#         else:
#             if np.isscalar(dones):
#                 rewards = np.expand_dims(np.asarray(rewards), 0)
#                 dones = np.expand_dims(np.asarray(dones), 0)
#             return self.obs_to_torch(obs), torch.from_numpy(rewards), torch.from_numpy(dones), infos

#     def obs_to_torch(self, obs):
#         if isinstance(obs, dict):
#             if 'obs' in obs:
#                 obs = obs['obs']
#             if isinstance(obs, dict):
#                 upd_obs = {}
#                 for key, value in obs.items():
#                     upd_obs[key] = self._obs_to_tensors_internal(value, False)
#             else:
#                 upd_obs = self.cast_obs(obs)
#         else:
#             upd_obs = self.cast_obs(obs)
#         return upd_obs

#     def _obs_to_tensors_internal(self, obs, cast_to_dict=True):
#         if isinstance(obs, dict):
#             upd_obs = {}
#             for key, value in obs.items():
#                 upd_obs[key] = self._obs_to_tensors_internal(value, False)
#         else:
#             upd_obs = self.cast_obs(obs)
#         return upd_obs

#     def cast_obs(self, obs):
#         if isinstance(obs, torch.Tensor):
#             self.is_tensor_obses = True
#         elif isinstance(obs, np.ndarray):
#             assert(obs.dtype != np.int8)
#             if obs.dtype == np.uint8:
#                 obs = torch.ByteTensor(obs).to(self.device)
#             else:
#                 obs = torch.FloatTensor(obs).to(self.device)
#         elif np.isscalar(obs):
#             obs = torch.FloatTensor([obs]).to(self.device)
#         return obs

#     def preprocess_actions(self, actions):
#         if not self.is_tensor_obses:
#             actions = actions.cpu().numpy()
#         return actions

#     def env_reset(self, env):
#         obs = env.reset()
#         return self.obs_to_torch(obs)

#     def restore(self, fn):
#         raise NotImplementedError('restore')

#     def get_weights(self):
#         weights = {}
#         weights['model'] = self.model.state_dict()
#         return weights

#     def set_weights(self, weights):
#         self.model.load_state_dict(weights['model'])
#         if self.normalize_input and 'running_mean_std' in weights:
#             self.model.running_mean_std.load_state_dict(weights['running_mean_std'])

#     def create_env(self):
#         return env_configurations.configurations[self.env_name]['env_creator'](**self.env_config)

#     def get_action(self, obs, is_determenistic=False):
#         raise NotImplementedError('step')

#     def get_masked_action(self, obs, mask, is_determenistic=False):
#         raise NotImplementedError('step')

#     def reset(self):
#         raise NotImplementedError('raise')

#     def init_rnn(self):
#         if self.is_rnn:
#             rnn_states = self.model.get_default_rnn_state()
#             self.states = [torch.zeros((s.size()[0], self.batch_size, s.size(
#             )[2]), dtype=torch.float32).to(self.device) for s in rnn_states]

#     # def run(self):
#     #     n_games = self.games_num
#     #     render = self.render_env
#     #     n_game_life = self.n_game_life
#     #     is_deterministic = self.is_deterministic
#     #     sum_rewards = 0
#     #     sum_steps = 0
#     #     sum_game_res = 0
#     #     n_games = n_games * n_game_life
#     #     games_played = 0
#     #     has_masks = False
#     #     has_masks_func = getattr(self.env, "has_action_mask", None) is not None

#     #     op_agent = getattr(self.env, "create_agent", None)
#     #     if op_agent:
#     #         agent_inited = True
#     #         # print('setting agent weights for selfplay')
#     #         # self.env.create_agent(self.env.config)
#     #         # self.env.set_weights(range(8),self.get_weights())

#     #     if has_masks_func:
#     #         has_masks = self.env.has_action_mask()

#     #     self.wait_for_checkpoint()

#     #     need_init_rnn = self.is_rnn
#     #     total_successes = 0
#     #     total_dones = 0
#     #     print(f"max_steps: {self.max_steps}")
#     #     print(f"n_games: {n_games}")
#     #     n_games = 50
#     #     total_map = 0
#     #     for _ in range(n_games):
#     #         if games_played >= n_games:
#     #             break

#     #         obses = self.env_reset(self.env)
#     #         batch_size = 1
#     #         batch_size = self.get_batch_size(obses, batch_size)

#     #         if need_init_rnn:
#     #             self.init_rnn()
#     #             need_init_rnn = False

#     #         cr = torch.zeros(batch_size, dtype=torch.float32)
#     #         steps = torch.zeros(batch_size, dtype=torch.float32)

#     #         print_game_res = False

#     #         for n in range(self.max_steps):
#     #             if games_played >= n_games:
#     #                     break
#     #             # print(f"games_played: {games_played} total_map: {total_map}")

#     #             # if games_played % 10 == 0 and total_map < 5:
#     #             #     print(f"games_played: {games_played} total_map: {total_map}")
#     #             #     self.env.env.random_all_map('_test', if_random=False, follow_order = total_map)
#     #             #     self.env.env.reset_jackal()

#     #             #     obses = self.env_reset(self.env)
#     #             #     # self.
#     #             #     total_map += 1
                    
#     #             #     print(f"test: {n} map session: {total_map}")        # print(f"total_successes: {total_successes}")

#     #             if self.evaluation and n % self.update_checkpoint_freq == 0:
#     #                 self.maybe_load_new_checkpoint()

#     #             if has_masks:
#     #                 masks = self.env.get_action_mask()
#     #                 action = self.get_masked_action(
#     #                     obses, masks, is_deterministic)
#     #             else:
#     #                 action = self.get_action(obses, is_deterministic)

#     #             obses, r, done, info = self.env_step(self.env, action)
#     #             # print(f"info['success]: {info['success']}")
#     #             successes = sum(info["success"])
#     #             done_infos = sum(done)
#     #             # print(f'successes: {successes}')
#     #             total_successes += successes
#     #             total_dones += done_infos
#     #             cr += r
#     #             steps += 1

#     #             if render:
#     #                 self.env.render(mode='human')
#     #                 time.sleep(self.render_sleep)

#     #             all_done_indices = done.nonzero(as_tuple=False)
#     #             done_indices = all_done_indices[::self.num_agents]
#     #             done_count = len(done_indices)
#     #             games_played += done_count

#     #             if done_count > 0:
#     #                 if self.is_rnn:
#     #                     for s in self.states:
#     #                         s[:, all_done_indices, :] = s[:,
#     #                                                       all_done_indices, :] * 0.0

#     #                 cur_rewards = cr[done_indices].sum().item()
#     #                 cur_steps = steps[done_indices].sum().item()

#     #                 cr = cr * (1.0 - done.float())
#     #                 steps = steps * (1.0 - done.float())
#     #                 sum_rewards += cur_rewards
#     #                 sum_steps += cur_steps

#     #                 game_res = 0.0
#     #                 if isinstance(info, dict):
#     #                     if 'battle_won' in info:
#     #                         print_game_res = True
#     #                         game_res = info.get('battle_won', 0.5)
#     #                     if 'scores' in info:
#     #                         print_game_res = True
#     #                         game_res = info.get('scores', 0.5)

#     #                 if self.print_stats:
#     #                     cur_rewards_done = cur_rewards/done_count
#     #                     cur_steps_done = cur_steps/done_count
#     #                     if print_game_res:
#     #                         print(f'reward: {cur_rewards_done:.2f} steps: {cur_steps_done:.1f} w: {game_res}')
#     #                     else:
#     #                         print(f'reward: {cur_rewards_done:.2f} steps: {cur_steps_done:.1f}')

#     #                 sum_game_res += game_res
#     #                 if games_played >= n_games:
#     #                     break
#     #                 # if batch_size//self.num_agents == 1 or games_played >= n_games:
#     #                 #     break
#     #                 # print(f"games_played: {games_played}")

#     #     # print(f"total_dones: {total_dones}")
#     #     # wandb.log({'Eval/success_rate': total_successes / total_dones * 100})
#     #     print(f"success_rate: {total_successes / total_dones * 100}")
#     #     print(sum_rewards)
#     #     if print_game_res:
#     #         print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps /
#     #               games_played * n_game_life, 'winrate:', sum_game_res / games_played * n_game_life)
#     #     else:
#     #         avg_reward = sum_rewards / games_played * n_game_life
#     #         avg_steps = sum_steps / games_played * n_game_life
#     #         print('av reward:', sum_rewards / games_played * n_game_life,
#     #               'av steps:', sum_steps / games_played * n_game_life)
#     #         return avg_reward, avg_steps
#     def run(self):
#         n_games = self.games_num
#         render = self.render_env
#         n_game_life = self.n_game_life
#         is_determenistic = self.is_determenistic
#         sum_rewards = 0
#         sum_steps = 0
#         sum_game_res = 0
#         n_games = n_games * n_game_life
#         games_played = 0
#         has_masks = False
#         has_masks_func = getattr(self.env, "has_action_mask", None) is not None
#         total_successes = 0
#         total_dones = 0
#         op_agent = getattr(self.env, "create_agent", None)
#         if op_agent:
#             agent_inited = True
#             #print('setting agent weights for selfplay')
#             # self.env.create_agent(self.env.config)
#             # self.env.set_weights(range(8),self.get_weights())

#         if has_masks_func:
#             has_masks = self.env.has_action_mask()

#         need_init_rnn = self.is_rnn
#         change_env_freq = n_games // 4
#         # map_order = 
#         # self.env.reset_cylinder_to_map(map_folder='worlds', if_random=False, if_order = 0)

#         for _ in range(n_games):
#             if games_played >= n_games:
#                 break

            
#             obses = self.env_reset(self.env)
#             batch_size = 1
#             batch_size = self.get_batch_size(obses, batch_size)

#             if need_init_rnn:
#                 self.init_rnn()
#                 need_init_rnn = False

#             cr = torch.zeros(batch_size, dtype=torch.float32)
#             steps = torch.zeros(batch_size, dtype=torch.float32)

#             print_game_res = False

#             for n in range(self.max_steps):
#                 if has_masks:
#                     masks = self.env.get_action_mask()
#                     action = self.get_masked_action(
#                         obses, masks, is_determenistic)
#                 else:
#                     action = self.get_action(obses, is_determenistic)

#                 obses, r, done, info = self.env_step(self.env, action)
#                 cr += r
#                 steps += 1
#                 successes = sum(info["success"])
#                 done_infos = sum(done)
#                 # print(f'successes: {successes}')
#                 total_successes += successes
#                 total_dones += done_infos
#                 if render:
#                     self.env.render(mode='human')
#                     time.sleep(self.render_sleep)

#                 all_done_indices = done.nonzero(as_tuple=False)
#                 done_indices = all_done_indices[::self.num_agents]
#                 done_count = len(done_indices)
#                 games_played += done_count
#                 if done_count > 0:
#                     if self.is_rnn:
#                         for s in self.states:
#                             s[:, all_done_indices, :] = s[:,all_done_indices, :] * 0.0

#                     cur_rewards = cr[done_indices].sum().item()
#                     cur_steps = steps[done_indices].sum().item()

#                     cr = cr * (1.0 - done.float())
#                     steps = steps * (1.0 - done.float())
#                     sum_rewards += cur_rewards
#                     sum_steps += cur_steps

#                     game_res = 0.0
#                     if isinstance(info, dict):
#                         if 'battle_won' in info:
#                             print_game_res = True
#                             game_res = info.get('battle_won', 0.5)
#                         if 'scores' in info:
#                             print_game_res = True
#                             game_res = info.get('scores', 0.5)

#                     if self.print_stats:
#                         if print_game_res:
#                             print('reward:', cur_rewards/done_count,
#                                   'steps:', cur_steps/done_count, 'w:', game_res)
#                         else:
#                             print('reward:', cur_rewards/done_count,
#                                   'steps:', cur_steps/done_count)
#                     print(f"games_played/n_games: {games_played/n_games}")

#                     sum_game_res += game_res
#                     if batch_size//self.num_agents == 1 or games_played >= n_games:
#                         break
#         print(f"success_rate: {total_successes / total_dones * 100}")

#         print(sum_rewards)
#         if print_game_res:
#             print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps /
#                   games_played * n_game_life, 'winrate:', sum_game_res / games_played * n_game_life)
#         else:
#             print('av reward:', sum_rewards / games_played * n_game_life,
#                   'av steps:', sum_steps / games_played * n_game_life)
#     # def run(self):
#     #     n_games = self.games_num * self.n_game_life
#     #     render = self.render_env
#     #     is_determenistic = self.is_determenistic
#     #     sum_rewards = 0
#     #     sum_steps = 0
#     #     sum_game_res = 0
#     #     games_played = 0
#     #     has_masks = False
#     #     has_masks_func = getattr(self.env, "has_action_mask", None) is not None
#     #     total_successes = 0
#     #     total_dones = 0
#     #     need_init_rnn = self.is_rnn

#     #     if has_masks_func:
#     #         has_masks = self.env.has_action_mask()

#     #     reset_intervals = set()  # Set to keep track of which intervals have had reset_cylinder_to_map called
#     #     change_env_freq = n_games // 5  # Determine the interval for resetting

#     #     for _ in range(n_games):
#     #         current_interval = games_played // change_env_freq
#     #         if current_interval not in reset_intervals:
#     #             self.env.reset_cylinder_to_map(map_folder='worlds', if_random=False, if_order = current_interval)
#     #             reset_intervals.add(current_interval)

#     #         obses = self.env_reset(self.env)
#     #         batch_size = self.get_batch_size(obses, 1)

#     #         if need_init_rnn:
#     #             self.init_rnn()
#     #             need_init_rnn = False

#     #         cr = torch.zeros(batch_size, dtype=torch.float32)
#     #         steps = torch.zeros(batch_size, dtype=torch.float32)
#     #         print_game_res = False

#     #         for n in range(self.max_steps):
#     #             # action = self.get_action(obses, is_determenistic, has_masks)
#     #             if has_masks:
#     #                 masks = self.env.get_action_mask()
#     #                 action = self.get_masked_action(
#     #                     obses, masks, is_determenistic)
#     #             else:
#     #                 action = self.get_action(obses, is_determenistic)

#     #             obses, r, done, info = self.env_step(self.env, action)
#     #             cr += r
#     #             steps += 1
#     #             successes, done_infos = self.update_success_done(info, done)
#     #             total_successes += successes
#     #             total_dones += done_infos

#     #             if render:
#     #                 self.env.render(mode='human')
#     #                 time.sleep(self.render_sleep)

#     #             games_played, done_indices = self.update_game_played(done)
#     #             if done_indices.size(0) > 0:
#     #                 cur_rewards, cur_steps = self.update_rewards_steps(cr, steps, done_indices)
#     #                 sum_rewards += cur_rewards
#     #                 sum_steps += cur_steps
#     #                 game_res = self.get_game_result(info, print_game_res)

#     #                 sum_game_res += game_res
#     #                 cr, steps = self.reset_cr_steps(cr, steps, done)

#     #                 if self.print_stats:
#     #                     # self.print_game_statistics(cur_rewards, cur_steps, game_res, done_indices.size(0))
#     #                     pass
#     #                 if games_played >= n_games:
#     #                     break
#     #                 print(f"total_successes: {total_successes} total_dones: {total_dones} success_rate: {total_successes / total_dones * 100}")
#     #                 self.print_final_statistics(total_successes, total_dones, sum_rewards, sum_steps, sum_game_res, games_played)

#     # def get_action(self, obses, is_determenistic, has_masks):
#     #     if has_masks:
#     #         masks = self.env.get_action_mask()
#     #         return self.get_masked_action(obses, masks, is_determenistic)
#     #     else:
#     #         return self.get_action(obses, is_determenistic)

#     def update_success_done(self, info, done):
#         successes = sum(info["success"])
#         done_infos = sum(done)
#         if done_infos > 0:
#             print(f"successes: {successes} done_infos: {done_infos}, success_rate: {successes / done_infos * 100}")
#         return successes, done_infos

#     def update_game_played(self, done):
#         all_done_indices = done.nonzero(as_tuple=False)
#         done_indices = all_done_indices[::self.num_agents]
#         games_played = len(done_indices)
#         return games_played, done_indices

#     def update_rewards_steps(self, cr, steps, done_indices):
#         cur_rewards = cr[done_indices].sum().item()
#         cur_steps = steps[done_indices].sum().item()
#         return cur_rewards, cur_steps

#     def get_game_result(self, info, print_game_res):
#         game_res = 0.0
#         if isinstance(info, dict) and ('battle_won' in info or 'scores' in info):
#             print_game_res = True
#             game_res = info.get('battle_won', info.get('scores', 0.5))
#         return game_res

#     def reset_cr_steps(self, cr, steps, done):
#         cr = cr * (1.0 - done.float())
#         steps = steps * (1.0 - done.float())
#         return cr, steps

#     def print_game_statistics(self, cur_rewards, cur_steps, game_res, done_count):
#         print(f"reward: {cur_rewards/done_count}, steps: {cur_steps/done_count}, winrate: {game_res}")

#     def print_final_statistics(self, total_successes, total_dones, sum_rewards, sum_steps, sum_game_res, games_played):
#         print(f"success_rate: {total_successes / total_dones * 100}")
#         print(f"average reward: {sum_rewards / games_played}, average steps: {sum_steps / games_played}, win rate: {sum_game_res / games_played}")

#     def get_batch_size(self, obses, batch_size):
#         obs_shape = self.obs_shape
#         if type(self.obs_shape) is dict:
#             if 'obs' in obses:
#                 obses = obses['obs']
#             keys_view = self.obs_shape.keys()
#             keys_iterator = iter(keys_view)
#             if 'observation' in obses:
#                 first_key = 'observation'
#             else:
#                 first_key = next(keys_iterator)
#             obs_shape = self.obs_shape[first_key]
#             obses = obses[first_key]

#         if len(obses.size()) > len(obs_shape):
#             batch_size = obses.size()[0]
#             self.has_batch_dimension = True

#         self.batch_size = batch_size

#         return batch_size


import os
import shutil
import threading
import time
import gym
import numpy as np
import torch
import copy
from os.path import basename
from typing import Optional
from rl_games.common import vecenv
from rl_games.common import env_configurations
from rl_games.algos_torch import model_builder
import wandb

def create_env(params):
    env_name = params['config']['env_name']
    env_config = params['config'].get('env_config', {})
    env = env_configurations.configurations[env_name]['env_creator'](**env_config)
    return env

class BasePlayer(object):

    def __init__(self, params, created_env=None, seed=None):
        self.config = config = params['config']
        self.load_networks(params)
        self.env_name = self.config['env_name']
        self.player_config = self.config.get('player', {})
        self.env_config = self.config.get('env_config', {})
        self.env_config = self.player_config.get('env_config', self.env_config)
        self.env_info = self.config.get('env_info')
        self.clip_actions = config.get('clip_actions', True)
        self.seed = self.env_config.pop('seed', None)



        if self.env_info is None:
            use_vecenv = self.player_config.get('use_vecenv', False)
            if created_env is None:
                if use_vecenv:
                    print('[BasePlayer] Creating vecenv: ', self.env_name)
                    self.env = vecenv.create_vec_env(
                        self.env_name, self.config['num_actors'], **self.env_config)
                    self.env_info = self.env.get_env_info()
                else:
                    print('[BasePlayer] Creating regular env: ', self.env_name)
                    self.env = self.create_env()
                    self.env_info = env_configurations.get_env_info(self.env)
            else:
                print('[BasePlayer] Using provided env: ', self.env_name)
                self.env = created_env
                # self.env_info = env_configurations.get_env_info(self.env)
                self.env_info = self.env.get_env_info()
        else:
            self.env = config.get('vec_env')

        self.num_agents = self.env_info.get('agents', 1)
        self.value_size = self.env_info.get('value_size', 1)
        self.action_space = self.env_info['action_space']

        self.observation_space = self.env_info['observation_space']
        if isinstance(self.observation_space, gym.spaces.Dict):
            self.obs_shape = {}
            for k, v in self.observation_space.spaces.items():
                self.obs_shape[k] = v.shape
        else:
            self.obs_shape = self.observation_space.shape
        self.is_tensor_obses = False

        self.states = None
        self.player_config = self.config.get('player', {})
        self.use_cuda = True
        self.batch_size = 1
        self.has_batch_dimension = False
        self.has_central_value = self.config.get(
            'central_value_config') is not None
        self.device_name = self.config.get('device_name', 'cuda')
        self.render_env = self.player_config.get('render', False)
        self.games_num = self.player_config.get('games_num', 10)

        if 'deterministic' in self.player_config:
            self.is_deterministic = self.player_config['deterministic']
        else:
            self.is_deterministic = self.player_config.get(
                'deterministic', True)

        self.n_game_life = self.player_config.get('n_game_life', 1)
        self.print_stats = self.player_config.get('print_stats', True)
        self.render_sleep = self.player_config.get('render_sleep', 0.002)
        self.max_steps = 108000 // 4
        # self.max_steps =
        self.device = torch.device(self.device_name)

        self.evaluation = self.player_config.get("evaluation", False)
        self.update_checkpoint_freq = self.player_config.get("update_checkpoint_freq", 100)
        # if we run player as evaluation worker this will take care of loading new checkpoints
        self.dir_to_monitor = self.player_config.get("dir_to_monitor")
        # path to the newest checkpoint
        self.checkpoint_to_load: Optional[str] = None

        if self.evaluation and self.dir_to_monitor is not None:
            self.checkpoint_mutex = threading.Lock()
            self.eval_checkpoint_dir = os.path.join(self.dir_to_monitor, "eval_checkpoints")
            os.makedirs(self.eval_checkpoint_dir, exist_ok=True)

            patterns = ["*.pth"]
            from watchdog.observers import Observer
            from watchdog.events import PatternMatchingEventHandler
            self.file_events = PatternMatchingEventHandler(patterns)
            self.file_events.on_created = self.on_file_created
            self.file_events.on_modified = self.on_file_modified

            self.file_observer = Observer()
            self.file_observer.schedule(self.file_events, self.dir_to_monitor, recursive=False)
            self.file_observer.start()

    def wait_for_checkpoint(self):
        if self.dir_to_monitor is None:
            return

        attempt = 0
        while True:
            attempt += 1
            with self.checkpoint_mutex:
                if self.checkpoint_to_load is not None:
                    if attempt % 10 == 0:
                        print(f"Evaluation: waiting for new checkpoint in {self.dir_to_monitor}...")
                    break
            time.sleep(1.0)

        print(f"Checkpoint {self.checkpoint_to_load} is available!")

    def maybe_load_new_checkpoint(self):
        # lock mutex while loading new checkpoint
        with self.checkpoint_mutex:
            if self.checkpoint_to_load is not None:
                print(f"Evaluation: loading new checkpoint {self.checkpoint_to_load}...")
                # try if we can load anything from the pth file, this will quickly fail if the file is corrupted
                # without triggering the retry loop in "safe_filesystem_op()"
                load_error = False
                try:
                    torch.load(self.checkpoint_to_load)
                except Exception as e:
                    print(f"Evaluation: checkpoint file is likely corrupted {self.checkpoint_to_load}: {e}")
                    load_error = True

                if not load_error:
                    try:
                        self.restore(self.checkpoint_to_load)
                    except Exception as e:
                        print(f"Evaluation: failed to load new checkpoint {self.checkpoint_to_load}: {e}")

                # whether we succeeded or not, forget about this checkpoint
                self.checkpoint_to_load = None

    

    def process_new_eval_checkpoint(self, path):
        with self.checkpoint_mutex:
            # print(f"New checkpoint {path} available for evaluation")
            # copy file to eval_checkpoints dir using shutil
            # since we're running the evaluation worker in a separate process,
            # there is a chance that the file is changed/corrupted while we're copying it
            # not sure what we can do about this. In practice it never happened so far though
            try:
                eval_checkpoint_path = os.path.join(self.eval_checkpoint_dir, basename(path))
                shutil.copyfile(path, eval_checkpoint_path)
            except Exception as e:
                print(f"Failed to copy {path} to {eval_checkpoint_path}: {e}")
                return

            self.checkpoint_to_load = eval_checkpoint_path

    def on_file_created(self, event):
        self.process_new_eval_checkpoint(event.src_path)

    def on_file_modified(self, event):
        self.process_new_eval_checkpoint(event.src_path)

    def load_networks(self, params):
        builder = model_builder.ModelBuilder()
        self.config['network'] = builder.load(params)

    def _preproc_obs(self, obs_batch):
        if type(obs_batch) is dict:
            obs_batch = copy.copy(obs_batch)
            for k, v in obs_batch.items():
                if v.dtype == torch.uint8:
                    obs_batch[k] = v.float() / 255.0
                else:
                    obs_batch[k] = v
        else:
            if obs_batch.dtype == torch.uint8:
                obs_batch = obs_batch.float() / 255.0
        return obs_batch

    def env_step(self, env, actions):
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()
        obs, rewards, dones, infos = env.step(actions)
        if hasattr(obs, 'dtype') and obs.dtype == np.float64:
            obs = np.float32(obs)
        if self.value_size > 1:
            rewards = rewards[0]
        if self.is_tensor_obses:
            return self.obs_to_torch(obs), rewards.cpu(), dones.cpu(), infos
        else:
            if np.isscalar(dones):
                rewards = np.expand_dims(np.asarray(rewards), 0)
                dones = np.expand_dims(np.asarray(dones), 0)
            return self.obs_to_torch(obs), torch.from_numpy(rewards), torch.from_numpy(dones), infos

    def obs_to_torch(self, obs):
        if isinstance(obs, dict):
            if 'obs' in obs:
                obs = obs['obs']
            if isinstance(obs, dict):
                upd_obs = {}
                for key, value in obs.items():
                    upd_obs[key] = self._obs_to_tensors_internal(value, False)
            else:
                upd_obs = self.cast_obs(obs)
        else:
            upd_obs = self.cast_obs(obs)
        return upd_obs

    def _obs_to_tensors_internal(self, obs, cast_to_dict=True):
        if isinstance(obs, dict):
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self._obs_to_tensors_internal(value, False)
        else:
            upd_obs = self.cast_obs(obs)
        return upd_obs

    def cast_obs(self, obs):
        if isinstance(obs, torch.Tensor):
            self.is_tensor_obses = True
        elif isinstance(obs, np.ndarray):
            assert (obs.dtype != np.int8)
            if obs.dtype == np.uint8:
                obs = torch.ByteTensor(obs).to(self.device)
            else:
                obs = torch.FloatTensor(obs).to(self.device)
        elif np.isscalar(obs):
            obs = torch.FloatTensor([obs]).to(self.device)
        return obs

    def preprocess_actions(self, actions):
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()
        return actions

    def env_reset(self, env):
        obs = env.reset()
        return self.obs_to_torch(obs)

    def restore(self, fn):
        raise NotImplementedError('restore')

    def get_weights(self):
        weights = {}
        weights['model'] = self.model.state_dict()
        return weights

    def set_weights(self, weights):
        self.model.load_state_dict(weights['model'])
        if self.normalize_input and 'running_mean_std' in weights:
            self.model.running_mean_std.load_state_dict(
                weights['running_mean_std'])

    def create_env(self):
        return env_configurations.configurations[self.env_name]['env_creator'](**self.env_config)

    def get_action(self, obs, is_deterministic=False):
        raise NotImplementedError('step')

    def get_masked_action(self, obs, mask, is_deterministic=False):
        raise NotImplementedError('step')

    def reset(self):
        raise NotImplementedError('raise')

    def init_rnn(self):
        if self.is_rnn:
            rnn_states = self.model.get_default_rnn_state()
            self.states = [torch.zeros((s.size()[0], self.batch_size, s.size(
            )[2]), dtype=torch.float32).to(self.device) for s in rnn_states]

    def run(self):
        n_games = self.games_num
        render = self.render_env
        n_game_life = self.n_game_life
        is_deterministic = self.is_deterministic
        sum_rewards = 0
        sum_steps = 0
        sum_game_res = 0
        n_games = n_games * n_game_life
        games_played = 0
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None

        op_agent = getattr(self.env, "create_agent", None)
        if op_agent:
            agent_inited = True
            # print('setting agent weights for selfplay')
            # self.env.create_agent(self.env.config)
            # self.env.set_weights(range(8),self.get_weights())

        if has_masks_func:
            has_masks = self.env.has_action_mask()

        self.wait_for_checkpoint()

        need_init_rnn = self.is_rnn
        total_successes = 0
        total_dones = 0
        print(f"max_steps: {self.max_steps}")
        print(f"n_games: {n_games}")
        n_games = 50
        change_map_freq = n_games//5
        total_map = 0
        for _ in range(n_games):
            if games_played >= n_games:
                break

            obses = self.env_reset(self.env)
            batch_size = 1
            batch_size = self.get_batch_size(obses, batch_size)

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            cr = torch.zeros(batch_size, dtype=torch.float32)
            steps = torch.zeros(batch_size, dtype=torch.float32)

            print_game_res = False

            for n in range(self.max_steps):
                if games_played >= n_games:
                        break
                # print(f"games_played: {games_played} total_map: {total_map}")

                # if games_played % 10 == 0 and total_map < 5:
                #     print(f"games_played: {games_played} total_map: {total_map}")
                #     self.env.env.random_all_map('_test', if_random=False, follow_order = total_map)
                #     self.env.env.reset_jackal()

                #     obses = self.env_reset(self.env)
                #     # self.
                #     total_map += 1
                    
                #     print(f"test: {n} map session: {total_map}")        # print(f"total_successes: {total_successes}")

                if self.evaluation and n % self.update_checkpoint_freq == 0:
                    self.maybe_load_new_checkpoint()

                if has_masks:
                    masks = self.env.get_action_mask()
                    action = self.get_masked_action(
                        obses, masks, is_deterministic)
                else:
                    action = self.get_action(obses, is_deterministic)

                obses, r, done, info = self.env_step(self.env, action)
                # print(f"info['success]: {info['success']}")
                successes = sum(info["success"])
                done_infos = sum(done)
                # print(f'successes: {successes}')
                total_successes += successes
                total_dones += done_infos
                cr += r
                steps += 1

                if render:
                    self.env.render(mode='human')
                    time.sleep(self.render_sleep)

                all_done_indices = done.nonzero(as_tuple=False)
                done_indices = all_done_indices[::self.num_agents]
                done_count = len(done_indices)
                if done_count > 0 and games_played % change_map_freq == 0:
                    self.env.env.random_all_map('_eval', if_random=True)
                    obses = self.env_reset(self.env)
                    # pass
                games_played += done_count

                if done_count > 0:
                    if self.is_rnn:
                        for s in self.states:
                            s[:, all_done_indices, :] = s[:,
                                                          all_done_indices, :] * 0.0

                    cur_rewards = cr[done_indices].sum().item()
                    cur_steps = steps[done_indices].sum().item()

                    cr = cr * (1.0 - done.float())
                    steps = steps * (1.0 - done.float())
                    sum_rewards += cur_rewards
                    sum_steps += cur_steps

                    game_res = 0.0
                    if isinstance(info, dict):
                        if 'battle_won' in info:
                            print_game_res = True
                            game_res = info.get('battle_won', 0.5)
                        if 'scores' in info:
                            print_game_res = True
                            game_res = info.get('scores', 0.5)

                    if self.print_stats:
                        cur_rewards_done = cur_rewards/done_count
                        cur_steps_done = cur_steps/done_count
                        if print_game_res:
                            print(f'reward: {cur_rewards_done:.2f} steps: {cur_steps_done:.1f} w: {game_res}')
                        else:
                            print(f'reward: {cur_rewards_done:.2f} steps: {cur_steps_done:.1f}')

                    sum_game_res += game_res
                    if games_played >= n_games:
                        break
                    # if batch_size//self.num_agents == 1 or games_played >= n_games:
                    #     break
                    # print(f"games_played: {games_played}")

        # print(f"total_dones: {total_dones}")
        wandb.log({'Eval/success_rate': total_successes / total_dones * 100})
        print(f"success_rate: {total_successes / total_dones * 100}")
        print(sum_rewards)
        if print_game_res:
            print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps /
                  games_played * n_game_life, 'winrate:', sum_game_res / games_played * n_game_life)
        else:
            avg_reward = sum_rewards / games_played * n_game_life
            avg_steps = sum_steps / games_played * n_game_life
            print('av reward:', sum_rewards / games_played * n_game_life,
                  'av steps:', sum_steps / games_played * n_game_life)
            return avg_reward, avg_steps

    def get_batch_size(self, obses, batch_size):
        obs_shape = self.obs_shape
        if type(self.obs_shape) is dict:
            if 'obs' in obses:
                obses = obses['obs']
            keys_view = self.obs_shape.keys()
            keys_iterator = iter(keys_view)
            if 'observation' in obses:
                first_key = 'observation'
            else:
                first_key = next(keys_iterator)
            obs_shape = self.obs_shape[first_key]
            obses = obses[first_key]

        if len(obses.size()) > len(obs_shape):
            batch_size = obses.size()[0]
            self.has_batch_dimension = True

        self.batch_size = batch_size

        return batch_size
