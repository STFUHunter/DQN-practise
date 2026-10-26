from collections import deque
import itertools
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import time

GAMMA = 0.99
BATCH_SIZE = 32
BUFFER_SIZE = 50000
MIN_REPLAY_SIZE = 1000
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 40000
TARGET_UPDATE_FREQ = 1000

env = gym.make(
    'FrozenLake-v1',
    desc=None,
    map_name="4x4",
    is_slippery=True
)

num_states = env.observation_space.n
num_actions = env.action_space.n

#since we have 16 inputs we need to one hot encode the state we are in
def state_to_oneHot(state, num_states=16):
    state_t = torch.tensor(state, dtype=torch.int64)
    one_hot = F.one_hot(state_t, num_classes=num_states).to(torch.float32)
    
    return one_hot

#the network
class DQN(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(env.observation_space.n, 32),
            nn.ReLU(),
            nn.Linear(32, env.action_space.n)
        )

    def forward(self, x):
        return self.net(x)

    def act(self, state):
        one_hot = state_to_oneHot(state)
        q_values = self(one_hot.unsqueeze(0))
        max_q_index = torch.argmax(q_values, dim=1)[0]

        return max_q_index.item()

#replay memory
memory = deque(maxlen=BUFFER_SIZE)
reward_buffer = deque([0.0], maxlen=100)

#initializing nets with env
online_net = DQN(env)
target_net = DQN(env)

#init episode reward as 0
episode_reward = 0.0

#load the state dict of online net into the target net
target_net.load_state_dict(online_net.state_dict())

#choosing the optimizer
optimizer = torch.optim.Adam(online_net.parameters(), lr=3e-4)

state, _ = env.reset()
for _ in range(MIN_REPLAY_SIZE):
    #compute action
    action = env.action_space.sample()
    
    #load the new state and other info (reward, terminated, truncated) by giving it an action
    new_state, reward, terminated, truncated, _ = env.step(action)
    
    #checking if the game is over
    done = terminated or truncated
    
    #get the transion as a tuple and then send it to memort
    transition = (state, action, reward, done, new_state)
    memory.append(transition)
    
    #update state
    state = new_state

    if done:
        state, _ = env.reset()

#training loop
state, _ = env.reset()

for step in itertools.count():
    #implementing epsilon
    epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])

    #take a random number and if it surpasses epsi take a random action else take the "best action"
    if random.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = online_net.act(state)

    #update and check if the game is over or not
    new_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    #update transition and append it to memory
    transition = (state, action, reward, done, new_state)
    memory.append(transition)
    state = new_state
    
    #add up rewards
    episode_reward += reward

    #check if game is done or not
    if done:
        #update state, append reward to reward buffer, 0 out the ep rew
        state, _ = env.reset()
        reward_buffer.append(episode_reward)
        episode_reward = 0.0
    
    #gradiant stepping sample from memory
    transitions = random.sample(memory, BATCH_SIZE)

    #give each var its value
    states = np.asarray([t[0] for t in transitions])
    actions = np.asarray([t[1] for t in transitions])
    rewards = np.asarray([t[2] for t in transitions])
    dones = np.asarray([t[3] for t in transitions])
    new_states = np.asarray([t[4] for t in transitions])

    #turn them to tensors and unsqueeze(0) every one except both states
    states_t = torch.as_tensor(states, dtype=torch.int64)
    actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
    rewards_t = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(-1)
    dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1)
    new_states_t = torch.as_tensor(new_states, dtype=torch.int64)

    #one hot encoding both states
    states_t_oneHot = F.one_hot(states_t, num_classes=num_states).to(torch.float32)
    new_states_t_oneHot = F.one_hot(new_states_t, num_classes=num_states).to(torch.float32)

    #compute targets
    target_q_values = target_net(new_states_t_oneHot)
    max_q_target_values = target_q_values.max(dim=1, keepdim=True)[0]
    targets = rewards_t + GAMMA * (1 - dones_t) * max_q_target_values

    #computing loss
    q_values = online_net(states_t_oneHot)
    action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)
    loss = F.smooth_l1_loss(action_q_values, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(online_net.state_dict())
    
    if step % 1000 == 0:
        print(f"Step {step} | Avg Rew: {np.mean(reward_buffer):.2f}")
        if len(reward_buffer) == 100 and np.mean(reward_buffer) >=0.72:
            print(f'Game was solved with {np.mean(reward_buffer)}')
            eval_env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=True, render_mode="human")
            for episode in range(5):  # Watch 5 evaluation episodes
                eval_state, _ = eval_env.reset()
                eval_done = False
                step_count = 0

                print(f"--- Visual Episode {episode + 1} ---")

                while not eval_done:
                    # 1. Purely greedy action (epsilon = 0)
                    action = online_net.act(eval_state)

                    # 2. Step the visual environment
                    eval_state, reward, terminated, truncated, _ = eval_env.step(action)
                    eval_done = terminated or truncated

                    step_count += 1
                    time.sleep(0.3)  # Slow down execution so steps are visible

                if reward == 1.0:
                    print(f"Result: Reached the Goal in {step_count} steps!")
                else:
                    print("Result: Fell into a Hole!")

                time.sleep(1.0)  # Brief pause between episodes

            eval_env.close()
            break  # Exit training loop