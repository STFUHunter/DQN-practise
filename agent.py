import torch
import itertools
import flappy_bird_gymnasium
import gymnasium as gym
from dqn import DQN
from experience_replay import MemoryReplay
import random
import yaml


class Agent():
    def __init__(self, hyper_param_set):
        with open('hyperparams.yml', 'r') as file:
            all_hyperparams_sets = yaml.safe_load(file)
            hyperparams = all_hyperparams_sets[hyper_param_set]

        self.replay_memory_size = hyperparams['replay_memory_size']
        self.mini_batch_size    = hyperparams['mini_batch_size']
        self.epsilon_init       = hyperparams['epsilon_init']
        self.epsilon_decay      = hyperparams['epsilon_decay']
        self.epsilon_min        = hyperparams['epsilon_min']

    def run(self, is_training=True, render=False):
        #env = gym.make("FlappyBird-v0", render_mode="human" if render else None)
        env = gym.make("CartPole-v1", render_mode="human" if render else None)
        num_state = env.observation_space.shape[0]
        num_action = env.action_space.n
        reward_per_episode = []
        epsilon_history = []
        policy_net = DQN(num_state, num_action)

        if is_training:
            memory = MemoryReplay(self.replay_memory_size)
            epsilon = self.epsilon_init

        for epoch in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float)
            terminated = False
            reward_episode = 0.0
            
            while not terminated:
                # Next action
                
                
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64)
                else:
                    with torch.no_grad():
                        action = policy_net(state.unsqueeze(dim=1)).squeeze().argmax()
                
                # Processing
                new_state, reward, terminated, _, info = env.step(action.item())
                
                #accumalate rewards
                reward_episode += reward

                new_state = torch.tensor(new_state, dtype=torch.float)
                reward = torch.tensor(reward, dtype=torch.float)

                #if we are training we want to append memory after taking action
                if is_training:
                    memory.append((state, action, new_state, reward, terminated))

                #update state
                state = new_state

            reward_per_episode.append(reward_episode)
            epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
            epsilon_history.append(epsilon)

if __name__ == '__main__':
    agent = Agent('cartpool')
    agent.run(is_training=True, render=False)
