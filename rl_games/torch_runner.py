import os
import time
import numpy as np
import random
import copy
import torch
import yaml
from argparse import Namespace
import wandb
from rl_games import envs
from rl_games.common import object_factory
from rl_games.common import env_configurations
from rl_games.common import experiment
from rl_games.common import tr_helpers

from rl_games.algos_torch import model_builder
from rl_games.algos_torch import a2c_continuous
from rl_games.algos_torch import a2c_discrete
from rl_games.algos_torch import players
from rl_games.common.algo_observer import DefaultAlgoObserver
from rl_games.algos_torch import sac_agent
import rl_games.networks
from dcgan.testmodel import make_agent_env, make_agent_env_barn
from dcgan.init_agent import AdversarialAgentTrainer
from dcgan.evaluation_map import test_metrics_for_each_map
def _restore(agent, args):
    if 'checkpoint' in args and args['checkpoint'] is not None and args['checkpoint'] !='':
        agent.restore(args['checkpoint'])

def _override_sigma(agent, args):
    if 'sigma' in args and args['sigma'] is not None:
        net = agent.model.a2c_network
        if hasattr(net, 'sigma') and hasattr(net, 'fixed_sigma'):
            if net.fixed_sigma:
                with torch.no_grad():
                    net.sigma.fill_(float(args['sigma']))
            else:
                print('Print cannot set new sigma because fixed_sigma is False')
class Runner:
    def __init__(self, algo_observer=None):
        self.algo_factory = object_factory.ObjectFactory()
        self.algo_factory.register_builder('PAIR_Agent', lambda **kwargs : a2c_continuous.PAIR_Agent(**kwargs))

        self.algo_factory.register_builder('a2c_continuous', lambda **kwargs : a2c_continuous.A2CAgent(**kwargs))
        self.algo_factory.register_builder('a2c_discrete', lambda **kwargs : a2c_discrete.DiscreteA2CAgent(**kwargs)) 
        self.algo_factory.register_builder('sac', lambda **kwargs: sac_agent.SACAgent(**kwargs))
        #self.algo_factory.register_builder('dqn', lambda **kwargs : dqnagent.DQNAgent(**kwargs))

        self.player_factory = object_factory.ObjectFactory()
        self.player_factory.register_builder('a2c_continuous', lambda **kwargs : players.PpoPlayerContinuous(**kwargs))
        self.player_factory.register_builder('a2c_discrete', lambda **kwargs : players.PpoPlayerDiscrete(**kwargs))
        self.player_factory.register_builder('sac', lambda **kwargs : players.SACPlayer(**kwargs))
        #self.player_factory.register_builder('dqn', lambda **kwargs : players.DQNPlayer(**kwargs))

        self.algo_observer = algo_observer if algo_observer else DefaultAlgoObserver()
        torch.backends.cudnn.benchmark = True
        print(f"running torch_runner inside {os.getcwd()}")
        ### it didnot help for lots for openai gym envs anyway :(
        #torch.backends.cudnn.deterministic = True
        #torch.use_deterministic_algorithms(True)
    def reset(self):
        pass

    def load_config(self, params):
        self.seed = params.get('seed', None)
        if self.seed is None:
            self.seed = int(time.time())
        
        if params["config"].get('multi_gpu', False):
            self.seed += int(os.getenv("LOCAL_RANK", "0"))
        print(f"self.seed = {self.seed}")

        self.algo_params = params['algo']
        self.algo_name = self.algo_params['name']
        self.exp_config = None
        if self.seed:

            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)
            
            # deal with environment specific seed if applicable
            if 'env_config' in params['config']:
                if not 'seed' in params['config']['env_config']:
                    params['config']['env_config']['seed'] = self.seed
                else:
                    if params["config"].get('multi_gpu', False):
                        params['config']['env_config']['seed'] += int(os.getenv("LOCAL_RANK", "0"))

        config = params['config']
        config['reward_shaper'] = tr_helpers.DefaultRewardsShaper(**config['reward_shaper'])
        if 'features' not in config:
            config['features'] = {}
        config['features']['observer'] = self.algo_observer
        self.params = params

    def load(self, yaml_conf):
        self.default_config = yaml_conf['params']
        self.load_config(params=self.default_config)

    def run_train(self, args):
        print('Started to train')
        agent = self.algo_factory.create(self.algo_name, base_name='run', params=self.params)
        _restore(agent, args)
        _override_sigma(agent, args)
        agent.train()

    def BARN_train(self, args):
        adversarial_args = Namespace(
            use_gae=True,
            gamma=0.995,
            gae_lambda=0.95,
            seed=77,
            recurrent_arch='lstm',
            recurrent_agent=True,
            recurrent_adversary_env=False,
            recurrent_hidden_size=256,
            use_global_critic=False,
            lr=0.0001,
            num_steps=20,
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
        # agent, env = make_agent_env(adversarial_args)
        agent, env = make_agent_env_barn(adversarial_args)
        adversarial_trainer = AdversarialAgentTrainer(adversarial_args, agent, env )
        
        print('Started to train')
        print(f"self.algo_name = {self.algo_name}")
        algo_name = 'PAIR_Agent'
        PAIR_agents = self.algo_factory.create(self.algo_name, base_name='run', params=self.params)
        
        
        
        # minigrid = GridEnvironment()
        # adversary = Adversary()

        _restore(PAIR_agents.protagonist, args)
        _override_sigma(PAIR_agents.protagonist, args)
        _restore(PAIR_agents.protagonist, args)
        _override_sigma(PAIR_agents.protagonist, args)
        _restore(PAIR_agents.antagonist, args)
        _override_sigma(PAIR_agents.antagonist, args)
        _restore(PAIR_agents.antagonist, args)
        _override_sigma(PAIR_agents.antagonist, args)
        
        
        num_envs = PAIR_agents.num_envs
        print(f"num_envs = {num_envs}")
        # intilize the mini grid
        pair_id = 0
        asset_root = './.././assets'
        self.create_world_folder(asset_root, pair_id = 'pair')
        
        if pair_id != '':
            # self.minigrid_rollout(pair_id = pair_id, num_envs = num_envs,env= minigrid, adversary=adversary, num_adversary_rollout=225, minigrid=minigrid)
            rollout_info = adversarial_trainer.agent_rollout(pair_id)
            print(f"adversarial_trainer.agent.storage.obs['obs'].shape = {adversarial_trainer.agent.storage.obs['obs'].shape}")
            PAIR_agents.env.random_all_map('_barn')
            PAIR_agents.env.env.reset_jackal()
        else:
            
            PAIR_agents.env.random_all_map(pair_id)
        ifeval = True
        if_pair = True
        evaluate_map = True
        evaluate_map_every = 20
        eval_every = 20
        warm_up = 0
        num_epochs = 100
        update_adversary_every = 1
        total_train_epoch = 10000
        for i in range(1, total_train_epoch):
            if if_pair:
                if i > warm_up and (i % update_adversary_every == 0):
                    rollout_info = adversarial_trainer.agent_rollout(i)
                    print(f"adversarial_trainer.agent.storage.obs['obs'].shape = {adversarial_trainer.agent.storage.obs['obs'].shape}")

                    # avg_num_obstackles =self.minigrid_rollout(pair_id = i, num_envs = num_envs,env= minigrid, adversary=adversary, num_adversary_rollout=225, minigrid=minigrid)
                    # print(f"avg_num_obstackles = {avg_num_obstackles}")
                    # wandb.log({f"avg_num_obstackles": avg_num_obstackles})
                    # env_map = adversarial_trainer.env.sample_img
                    # images = wandb.Image(env_map, caption=f"PAIR_ID: {i}")
                        
                    # wandb.log({f"Generated Map": images})
                    if evaluate_map:
                        if i % evaluate_map_every == 0:
                            # evaluate_map_every = 1

                            PAIR_agents.env.random_all_map('_barn')
                            PAIR_agents.env.env.reset_jackal()
                            
                            # map_pth = PAIR_agents.env.env.grid_root
                            # print(f"map_pth = {map_pth}")
                            # map_metric = test_metrics_for_each_map(map_pth)
                            # wandb.log({f"PAIR/closest_wall": map_metric['closest_wall']})
                            # wandb.log({f"PAIR/avg_visibility": map_metric['avg_visibility']})
                            # wandb.log({f"PAIR/dispersion": map_metric['dispersion']})
                            # wandb.log({f"PAIR/characteristic_dimension": map_metric['characteristic_dimension']})
                            # wandb.log({f"PAIR/occupancy_rate": map_metric['occupancy_rate']})
                            # PAIR_agents.protagonist.writer.add_scalar('PAIR/closest_wall', map_metric['closest_wall'], i)
                            # PAIR_agents.protagonist.writer.add_scalar('PAIR/avg_visibility', map_metric['avg_visibility'], i)
                            # PAIR_agents.protagonist.writer.add_scalar('PAIR/dispersion', map_metric['dispersion'], i)
                            # PAIR_agents.protagonist.writer.add_scalar('PAIR/characteristic_dimension', map_metric['characteristic_dimension'], i)
                            # PAIR_agents.protagonist.writer.add_scalar('PAIR/occupancy_rate', map_metric['occupancy_rate'], i)
            else:
                PAIR_agents.env.random_all_map('')
                PAIR_agents.env.env.reset_jackal()

            
            self.reset_seed()
            
            protagonist_rewards, protagonist_mean_rewards, _ = PAIR_agents.protagonist.train()
            self.reset_seed()
            if if_pair:
                PAIR_agents.env.random_all_map('_barn')
                PAIR_agents.env.env.reset_jackal()
            antagonist_rewards, antagonist_mean_rewards, _ = PAIR_agents.antagonist.train()

            regret = torch.abs(protagonist_rewards - antagonist_rewards)
            # print(f"regret.shape = {regret}")
            mean_regret = torch.mean(regret).cpu().numpy()
            wandb.log({f"PAIR/regret": mean_regret})
            wandb.log({f"PAIR/protagonist_rewards": protagonist_mean_rewards})
            wandb.log({f"PAIR/antagonist_rewards": antagonist_mean_rewards})
            PAIR_agents.protagonist.writer.add_scalar('PAIR/regret', mean_regret, i)
            PAIR_agents.protagonist.writer.add_scalar('PAIR/protagonist_rewards', protagonist_mean_rewards, i)
            PAIR_agents.protagonist.writer.add_scalar('PAIR/antagonist_rewards', antagonist_mean_rewards, i)
            # print(f"{i}: protagonist_rewards = {protagonist_rewards}; antagonist_rewards = {antagonist_rewards}")
            print(f"{i} training ended")
            if if_pair:
                if i == warm_up:
                    print(f"warm up done, {protagonist_mean_rewards}, {antagonist_mean_rewards}")
                    PAIR_agents.protagonist.writer.add_scalar('warm_up_rewards', protagonist_mean_rewards, i)
                if i > warm_up and (i % update_adversary_every == 0):
                    update_stats = adversarial_trainer.update_agent(regret)
                    # train_adversary(adversary, num_epochs=num_epochs, regret=regret, storage= adversary.storage)
            if ifeval:
                if i % eval_every == 0:
                    # PAIR_agents.env.random_cy
                    # PAIR_agents.env.random_all_map('')
                    PAIR_agents.env.random_all_map('_eval')

                    PAIR_agents.env.env.reset_jackal()

                    PAIR_agents.player.restore(PAIR_agents.protagonist.check_point_pth)
                    _override_sigma(PAIR_agents.player, args)
                    avg_reward, avg_steps = PAIR_agents.player.run()
                    wandb.log({f"Eval/avg_reward": avg_reward})
                    wandb.log({f"Eval/avg_steps": avg_steps})
                    PAIR_agents.protagonist.writer.add_scalar('Eval/avg_reward', avg_reward, i)
                    PAIR_agents.protagonist.writer.add_scalar('Eval/avg_steps', avg_steps, i)
                    print(f"evaluation ended, avg_reward = {avg_reward}, avg_steps = {avg_steps}")
                    # eval_stats = adversarial_trainer.eval_agent()
                    # print(f"eval_stats = {eval_stats}")
                    # PAIR_agents.protagonist.writer.add_scalar('eval_stats', eval_stats, i)
                    # wandb.log({f"eval_stats": eval_stats})
                    # train_adversary(adversary, num_epochs=num_epochs, regret=regret, storage= adversary.storage)


    def VAE_train(self, args):
        adversarial_args = Namespace(
            use_gae=True,
            gamma=0.995,
            gae_lambda=0.95,
            seed=77,
            recurrent_arch='lstm',
            recurrent_agent=True,
            recurrent_adversary_env=False,
            recurrent_hidden_size=256,
            use_global_critic=False,
            lr=0.0001,
            num_steps=20,
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
            latent_dim = 20,
            )
        agent, env = make_agent_env(adversarial_args)
        # agent, env = make_agent_env_barn(adversarial_args)
        adversarial_trainer = AdversarialAgentTrainer(adversarial_args, agent, env )
        
        print('Started to train')
        print(f"self.algo_name = {self.algo_name}")
        algo_name = 'PAIR_Agent'
        PAIR_agents = self.algo_factory.create(self.algo_name, base_name='run', params=self.params)
        
        
        
        # minigrid = GridEnvironment()
        # adversary = Adversary()

        _restore(PAIR_agents.protagonist, args)
        _override_sigma(PAIR_agents.protagonist, args)
        _restore(PAIR_agents.protagonist, args)
        _override_sigma(PAIR_agents.protagonist, args)
        _restore(PAIR_agents.antagonist, args)
        _override_sigma(PAIR_agents.antagonist, args)
        _restore(PAIR_agents.antagonist, args)
        _override_sigma(PAIR_agents.antagonist, args)
        
        
        num_envs = PAIR_agents.num_envs
        # print(f"num_envs = {num_envs}")
        # # intilize the mini grid
        pair_id = 0
        # asset_root = './.././assets'
        # self.create_world_folder(asset_root, pair_id = 'pair')
        
        if pair_id != '':
            # self.minigrid_rollout(pair_id = pair_id, num_envs = num_envs,env= minigrid, adversary=adversary, num_adversary_rollout=225, minigrid=minigrid)
            rollout_info = adversarial_trainer.agent_rollout(pair_id)
            # print(f"adversarial_trainer.agent.storage.obs['obs'].shape = {adversarial_trainer.agent.storage.obs['obs'].shape}")
            # PAIR_agents.env.random_all_map('_barn')
            # PAIR_agents.env.env.reset_jackal()
        else:
            pass
            # PAIR_agents.env.random_all_map(pair_id)
        ifeval = True
        if_pair = True
        evaluate_map = True
        evaluate_map_every = 20
        eval_every = 20
        warm_up = 0
        num_epochs = 100
        update_adversary_every = 1
        total_train_epoch = 10000
        for i in range(1, total_train_epoch):
            if if_pair:
                if i > warm_up and (i % update_adversary_every == 0):
                    rollout_info = adversarial_trainer.agent_rollout(i)
                    # print(f"adversarial_trainer.agent.storage.obs['obs'].shape = {adversarial_trainer.agent.storage.obs['obs'].shape}")

                    # avg_num_obstackles =self.minigrid_rollout(pair_id = i, num_envs = num_envs,env= minigrid, adversary=adversary, num_adversary_rollout=225, minigrid=minigrid)
                    # print(f"avg_num_obstackles = {avg_num_obstackles}")
                    # wandb.log({f"avg_num_obstackles": avg_num_obstackles})
                    # env_map = adversarial_trainer.env.sample_img
                    # images = wandb.Image(env_map, caption=f"PAIR_ID: {i}")
                        
                    # wandb.log({f"Generated Map": images})
                    if evaluate_map:
                        if i % evaluate_map_every == 0:
                            # evaluate_map_every = 1

                            # PAIR_agents.env.random_all_map('_barn')
                            # PAIR_agents.env.env.reset_jackal()
                            pass
                            # map_pth = PAIR_agents.env.env.grid_root
                            # print(f"map_pth = {map_pth}")
                            # map_metric = test_metrics_for_each_map(map_pth)
                            # wandb.log({f"PAIR/closest_wall": map_metric['closest_wall']})
                            # wandb.log({f"PAIR/avg_visibility": map_metric['avg_visibility']})
                            # wandb.log({f"PAIR/dispersion": map_metric['dispersion']})
                            # wandb.log({f"PAIR/characteristic_dimension": map_metric['characteristic_dimension']})
                            # wandb.log({f"PAIR/occupancy_rate": map_metric['occupancy_rate']})
                            # PAIR_agents.protagonist.writer.add_scalar('PAIR/closest_wall', map_metric['closest_wall'], i)
                            # PAIR_agents.protagonist.writer.add_scalar('PAIR/avg_visibility', map_metric['avg_visibility'], i)
                            # PAIR_agents.protagonist.writer.add_scalar('PAIR/dispersion', map_metric['dispersion'], i)
                            # PAIR_agents.protagonist.writer.add_scalar('PAIR/characteristic_dimension', map_metric['characteristic_dimension'], i)
                            # PAIR_agents.protagonist.writer.add_scalar('PAIR/occupancy_rate', map_metric['occupancy_rate'], i)
            else:
                # PAIR_agents.env.random_all_map('')
                # PAIR_agents.env.env.reset_jackal()
                pass
            
            # self.reset_seed()
            
            protagonist_rewards, protagonist_mean_rewards, _ = PAIR_agents.protagonist.train()
            # self.reset_seed()
            if if_pair:
                # PAIR_agents.env.random_all_map('_barn')
                # PAIR_agents.env.env.reset_jackal()
                pass
            antagonist_rewards, antagonist_mean_rewards, _ = PAIR_agents.antagonist.train()

            regret = torch.abs(protagonist_rewards - antagonist_rewards)
            # print(f"regret.shape = {regret}")
            mean_regret = torch.mean(regret).cpu().numpy()
            wandb.log({f"PAIR/regret": mean_regret})
            wandb.log({f"PAIR/protagonist_rewards": protagonist_mean_rewards})
            wandb.log({f"PAIR/antagonist_rewards": antagonist_mean_rewards})
            PAIR_agents.protagonist.writer.add_scalar('PAIR/regret', mean_regret, i)
            PAIR_agents.protagonist.writer.add_scalar('PAIR/protagonist_rewards', protagonist_mean_rewards, i)
            PAIR_agents.protagonist.writer.add_scalar('PAIR/antagonist_rewards', antagonist_mean_rewards, i)
            # print(f"{i}: protagonist_rewards = {protagonist_rewards}; antagonist_rewards = {antagonist_rewards}")
            print(f"{i} training ended")
            if if_pair:
                if i == warm_up:
                    print(f"warm up done, {protagonist_mean_rewards}, {antagonist_mean_rewards}")
                    PAIR_agents.protagonist.writer.add_scalar('warm_up_rewards', protagonist_mean_rewards, i)
                if i > warm_up and (i % update_adversary_every == 0):
                    update_stats = adversarial_trainer.update_agent(regret)
                    # train_adversary(adversary, num_epochs=num_epochs, regret=regret, storage= adversary.storage)
            if ifeval:
                if i % eval_every == 0:
                    # PAIR_agents.env.random_cy
                    # PAIR_agents.env.random_all_map('')


                    pass
                    # PAIR_agents.env.random_all_map('_eval')

                    # PAIR_agents.env.env.reset_jackal()

                    # PAIR_agents.player.restore(PAIR_agents.protagonist.check_point_pth)
                    # _override_sigma(PAIR_agents.player, args)
                    # avg_reward, avg_steps = PAIR_agents.player.run()
                    # wandb.log({f"Eval/avg_reward": avg_reward})
                    # wandb.log({f"Eval/avg_steps": avg_steps})
                    # PAIR_agents.protagonist.writer.add_scalar('Eval/avg_reward', avg_reward, i)
                    # PAIR_agents.protagonist.writer.add_scalar('Eval/avg_steps', avg_steps, i)
                    # print(f"evaluation ended, avg_reward = {avg_reward}, avg_steps = {avg_steps}")
                    
                    
                    
                    # eval_stats = adversarial_trainer.eval_agent()
                    # print(f"eval_stats = {eval_stats}")
                    # PAIR_agents.protagonist.writer.add_scalar('eval_stats', eval_stats, i)
                    # wandb.log({f"eval_stats": eval_stats})
                    # train_adversary(adversary, num_epochs=num_epochs, regret=regret, storage= adversary.storage)





    def base_train(self, args):
        print('Started to train')
        agent = self.algo_factory.create(self.algo_name, base_name='run', params=self.params)
        _restore(agent.protagonist, args)
        _override_sigma(agent.protagonist, args)
        _restore(agent.antagonist, args)
        _override_sigma(agent.antagonist, args)
        num_epoch = 100000
        # for i in range(num_epoch):
        # agent.env.env.reset_cylinder_to_map(map_folder='worlds_train', if_random=False, if_order=0)
        agent.protagonist.train()
        # agent.antagonist.train()

    def run_play(self, args):
        print('Started to play')
        player = self.create_player()
        _restore(player, args)
        _override_sigma(player, args)
        player.run()

    def create_player(self):
        return self.player_factory.create(self.algo_name, params=self.params)

    def reset(self):
        pass

    def run(self, args):
        load_path = None

        if args['train']:
            # self.run_train(args)
            # self.base_train(args)
            self.VAE_train(args)

        elif args['play']:
            self.run_play(args)
        else:
            self.run_train(args)