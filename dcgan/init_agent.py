from argparse import Namespace
from dcgan.algos import PPO, RolloutStorage, ACAgent
# from algos import PPO
# from algos import RolloutStorage
# from algos import ACAgent
from dcgan.models import MultigridNetworkTaskEmbedSingleStepContinuous
import torch
import wandb
# Simulating command-line arguments
# args = Namespace(
#     use_gae=True,
#     gamma=0.995,
#     gae_lambda=0.95,
#     seed=77,
#     recurrent_arch='lstm',
#     recurrent_agent=True,
#     recurrent_adversary_env=True,
#     recurrent_hidden_size=256,
#     use_global_critic=False,
#     lr=0.0001,
#     num_steps=1,
#     num_processes=32,
#     ppo_epoch=5,
#     num_mini_batch=1,
#     entropy_coef=0.0,
#     value_loss_coef=0.5,
#     clip_param=0.2,
#     clip_value_loss=True,
#     adv_entropy_coef=0.0,
#     max_grad_norm=0.5,
#     algo='ppo',
#     ued_algo='paired',
#     use_plr=False,
#     handle_timelimits=True,
#     log_grad_norm=False,
#     eps = 1e-5,
#     device='cuda',
# )

def model_for_multigrid_agent(env):
    adversary_observation_space = env.adversary_observation_space
    adversary_action_space = env.adversary_action_space

    model = MultigridNetworkTaskEmbedSingleStepContinuous(
        observation_space=adversary_observation_space,
        action_space=adversary_action_space,
        scalar_fc=10)
    return model

def make_barn_agent(args, env, actor_critic, device='cpu'):
    use_proper_time_limits = hasattr(env, 'get_max_episode_steps') and env.get_max_episode_steps() is not None \
        and args.handle_timelimits

    algo = PPO(
        actor_critic=actor_critic,
        clip_param=args.clip_param,
        ppo_epoch=args.ppo_epoch,
        num_mini_batch=args.num_mini_batch,
        value_loss_coef=args.value_loss_coef,
        entropy_coef=args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm,
        clip_value_loss=args.clip_value_loss,
        log_grad_norm=args.log_grad_norm
    )

    storage = RolloutStorage(
        model=actor_critic,
        num_steps=args.num_steps,
        num_processes=args.num_processes,
        observation_space=env.adversary_observation_space,
        action_space=env.adversary_action_space,
        recurrent_hidden_state_size=args.recurrent_hidden_size,
        recurrent_arch=False,
        use_proper_time_limits=use_proper_time_limits,
        use_popart=vars(args).get('adv_use_popart', False)
    )

    agent = ACAgent(algo=algo, storage=storage).to(device)
    return agent    
    

def make_agent(args, env, device='cpu'):
    actor_critic = model_for_multigrid_agent(env)

    use_proper_time_limits = hasattr(env, 'get_max_episode_steps') and env.get_max_episode_steps() is not None \
        and args.handle_timelimits

    algo = PPO(
        actor_critic=actor_critic,
        clip_param=args.clip_param,
        ppo_epoch=args.ppo_epoch,
        num_mini_batch=args.num_mini_batch,
        value_loss_coef=args.value_loss_coef,
        entropy_coef=args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm,
        clip_value_loss=args.clip_value_loss,
        log_grad_norm=args.log_grad_norm
    )

    storage = RolloutStorage(
        model=actor_critic,
        num_steps=args.num_steps,
        num_processes=args.num_processes,
        observation_space=env.adversary_observation_space,
        action_space=env.adversary_action_space,
        recurrent_hidden_state_size=args.recurrent_hidden_size,
        recurrent_arch=False,
        use_proper_time_limits=use_proper_time_limits,
        use_popart=vars(args).get('adv_use_popart', False)
    )

    agent = ACAgent(algo=algo, storage=storage).to(device)
    return agent
def compute_final_return(agent, env):
    return 0


# class Trainer:
#     def __init__(self, agent, env):
#         self.agent = agent
#         self.env = env
#         self.n_episodes = 10
#         self.max_steps_per_episode = 10
#     def rollout(self):
        
        
def train(env, agent, n_episodes, max_steps_per_episode):
    """
    Train the agent in the given environment.

    :param env: The gym environment.
    :param agent: The ACAgent instance.
    :param n_episodes: Number of episodes to train for.
    :param max_steps_per_episode: Maximum steps in each episode.
    """
    for episode in range(n_episodes):
        obs = env.reset()
        # agent.storage.reset()
        # print(f"obs: {obs}")
        agent.storage.copy_obs_to_index(obs,0)

        recurrent_hidden_states = torch.zeros([1,  agent.algo.actor_critic.recurrent_hidden_state_size])
        masks = torch.zeros(1, 1)
        bad_masks = torch.zeros(1, 1)
        # level_seeds and cliffhanger_masks are optional and depend on your specific use case

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_dist, recurrent_hidden_states = agent.act(
                    obs, recurrent_hidden_states, masks)
                # action_log_prob = action_log_dist.gather(-1, action)  # Assuming discrete actions
                action_log_prob = action_log_dist
                # print(f"action_log_prob.shape: {action_log_prob.shape}")
                # print(f"action.shape: {action.shape}")
                # print(f"action: {action}")
                # print(f"action_log_prob: {action_log_prob}")
                # print(f"action_log_dist: {action_log_dist}")
            # Observe reward and next obs
            # next_obs, reward, done, infos = env.step(action.cpu().numpy())
            next_obs, reward, done, infos = env.step(action)
            # Handle any additional flags or data from infos
            # For example, handling cliffhangers or level replays

            # Convert done to masks and bad_masks
            masks.fill_(0.0 if done else 1.0)
            bad_masks.fill_(0.0 if 'truncated' in infos else 1.0)
            
            # If done then clean the history of observations.
            # masks = torch.FloatTensor(
            #     [[0.0] if done_ else [1.0] for done_ in done])
            # bad_masks = torch.FloatTensor(
            #     [[0.0] if 'truncated' in info.keys() else [1.0]
            #      for info in infos])
            # level_seeds and cliffhanger_masks logic goes here

            # Insert the results into the RolloutStorage
            agent.insert(obs, recurrent_hidden_states, action, action_log_prob, action_log_dist,
                         value, reward, masks, bad_masks)

            obs = next_obs

            if done:
                break

        # After collecting rollouts, compute returns
        with torch.no_grad():
            next_value = agent.get_value(next_obs, recurrent_hidden_states, masks)
        # agent.storage.replace_final_return(torch.tensor([0]))
        #-----------------------
        #Add one modlue to calculate the final return from outside
        #---------
        # Sepearte this part out
        # agent.storage.replace_final_return(torch.randn(1).to('cuda'))

        # agent.storage.replace_final_return(torch.randn((100,1,1)).to('cuda'))
        #-----------------------
        agent.storage.compute_returns(next_value,
                        args.use_gae,
                        args.gamma,
                        args.gae_lambda,)

        # Perform PPO update
        agent.update()

        # Logging and any additional updates

    print("Training completed.")
# Example usage:
# Assuming env and agent name ('agent' or 'adversary_agent') are defined
# agent = make_agent('agent', env, device='cpu')


class AdversarialAgentTrainer:
    def __init__(self, args, agent, env):
        self.args = args
        self.agent = agent
        self.env = env
        self.adversary_env_rollout_steps = args.num_steps
        self.is_training = args.train
        self.device = args.device

    def agent_rollout_no_recurrent(self, episode):
        obs = self.env.reset()
        self.agent.storage.copy_obs_to_index(obs, 0)

        # recurrent_hidden_states = torch.zeros([1, self.agent.algo.actor_critic.recurrent_hidden_state_size])
        masks = torch.zeros(1, 1)
        bad_masks = torch.zeros(1, 1)
        rollout_info = {}
        rollout_returns = []

        for step in range(self.adversary_env_rollout_steps):
            with torch.no_grad():
                obs_id = self.agent.storage.get_obs(step)
                value, action, action_log_dist = self.agent.act(
                    obs_id, masks)
                action_log_prob = action_log_dist

            obs, reward, done, infos = self.env.step(action,episode, step)

            masks.fill_(0.0 if done else 1.0)
            bad_masks.fill_(0.0 if 'truncated' in infos else 1.0)

            self.agent.storage.insert(obs, recurrent_hidden_states, action, action_log_prob, action_log_dist,
                                    value, reward, masks, bad_masks)

            if 'episode' in infos:
                rollout_returns.append(infos['episode']['r'])

            if done and step < self.adversary_env_rollout_steps - 1:
                obs = self.env.reset()  # Reset the environment
                self.agent.storage.copy_obs_to_index(obs, step + 1)  # Insert the new observation at the next index

        rollout_info['returns'] = rollout_returns
        # print(f"Rollout info: {rollout_info}")

        return rollout_info
    def agent_rollout(self, episode):
        obs = self.env.reset()
        self.agent.storage.copy_obs_to_index(obs, 0)

        recurrent_hidden_states = torch.zeros([1, self.agent.algo.actor_critic.recurrent_hidden_state_size])
        masks = torch.zeros(1, 1)
        bad_masks = torch.zeros(1, 1)
        rollout_info = {}
        rollout_returns = []
        all_infos = []
        for step in range(self.adversary_env_rollout_steps):
            with torch.no_grad():
                obs_id = self.agent.storage.get_obs(step)
                value, action, action_log_dist, recurrent_hidden_states = self.agent.act(
                    obs_id, self.agent.storage.get_recurrent_hidden_state(step), masks)
                action_log_prob = action_log_dist

            obs, reward, done, infos = self.env.step(action,episode, step)
            all_infos.append(infos)
            masks.fill_(0.0 if done else 1.0)
            bad_masks.fill_(0.0 if 'truncated' in infos else 1.0)

            self.agent.storage.insert(obs, recurrent_hidden_states, action, action_log_prob, action_log_dist,
                                    value, reward, masks, bad_masks)

            if 'episode' in infos:
                rollout_returns.append(infos['episode']['r'])

            if done and step < self.adversary_env_rollout_steps - 1:
                obs = self.env.reset()  # Reset the environment
                self.agent.storage.copy_obs_to_index(obs, step + 1)  # Insert the new observation at the next index

        rollout_info['returns'] = rollout_returns
        # print(f"Rollout info: {rollout_info}")

        # take the mean of all infos. 
        # info = {'fill_pct': fill_pct, 'Distance to Closest Obstacle': metric[0], 'Average Visibility': metric[1], 'Dispersion': metric[2], 'Characteristic Dimension': metric[3], 'Tortuosity': metric[4]}
        avg_infos = {}
        for infos in all_infos:
            for key in infos.keys():
                if key not in avg_infos.keys():
                    avg_infos[key] = infos[key]
                else:
                    avg_infos[key] += infos[key]
        for key in avg_infos.keys():
            avg_infos[key] /= len(all_infos)
        print(f"avg_infos: {avg_infos}")
        wandb.log(avg_infos)
        return rollout_info

    # def agent_rollout(self):
    #     obs = self.env.reset()
    #     self.agent.storage.copy_obs_to_index(obs, 0)

    #     recurrent_hidden_states = torch.zeros([1, self.agent.algo.actor_critic.recurrent_hidden_state_size])
    #     masks = torch.zeros(1, 1)
    #     bad_masks = torch.zeros(1, 1)
    #     rollout_info = {}
    #     rollout_returns = []

    #     for step in range(self.adversary_env_rollout_steps):
    #         with torch.no_grad():
    #             obs_id = self.agent.storage.get_obs(step)
    #             value, action, action_log_dist, recurrent_hidden_states = self.agent.act(
    #                 obs_id, self.agent.storage.get_recurrent_hidden_state(step), masks)
    #             action_log_prob = action_log_dist

    #         obs, reward, done, infos = self.env.step(action)

    #         masks.fill_(0.0 if done else 1.0)
    #         bad_masks.fill_(0.0 if 'truncated' in infos else 1.0)

    #         self.agent.storage.insert(obs, recurrent_hidden_states, action, action_log_prob, action_log_dist,
    #                                           value, reward, masks, bad_masks)

    #         if 'episode' in infos:
    #             rollout_returns.append(infos['episode']['r'])

    #         if done:
    #             break

    #     rollout_info['returns'] = rollout_returns
    #     return rollout_info

    def update_agent(self, regret, obs = None):
        # masks = torch.FloatTensor(
        #     [[0.0] if done_ else [1.0] for done_ in done])
        # bad_masks = torch.FloatTensor([[0.0] if 'truncated' in info.keys() else [1.0]])
        
        with torch.no_grad():
            obs_id = self.agent.storage.get_obs(-1)
            next_value = self.agent.get_value(obs_id, self.agent.storage.get_recurrent_hidden_state(-1), self.agent.storage.masks[-1]).detach()
        self.agent.storage.replace_final_return(regret.to('cuda'))
        if obs :
            self.agent.storage.replace_obs(obs.to('cuda'))
        self.agent.storage.compute_returns(next_value, self.args.use_gae, self.args.gamma, self.args.gae_lambda)

        value_loss, action_loss, dist_entropy, update_info = self.agent.update()

        update_stats = {
            'value_loss': value_loss,
            'action_loss': action_loss,
            'dist_entropy': dist_entropy,
            'update_info': update_info
        }
        # print(f"Update stats: {update_stats}")

        return update_stats

    def train(self, n_episodes):
        for episode in range(n_episodes):
            print(f"Episode: {episode}")
            rollout_info = self.agent_rollout(episode)
            
            regret = torch.randn((self.args.num_steps,1,1))
            print(f"regret.shape: {regret.shape}")
            update_stats = self.update_agent(regret)

            # Logging and any additional updates (omitted for brevity)
            print(f"Episode {episode + 1}/{n_episodes}: Rollout info: {rollout_info}, Update stats: {update_stats}")

        print("Training completed.")