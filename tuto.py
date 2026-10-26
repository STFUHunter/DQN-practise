import itertools
import random
from collections import deque
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

GAMMA = 0.99
BATCH_SIZE = 32
BUFFER_SIZE = 50000
MIN_REPLAY_SIZE = 1000
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 10000
TARGET_UPDATE_FREQ = 1000


class Network(nn.Module):
    def __init__(self, env):
        super().__init__()
        state_dim = int(np.prod(env.observation_space.shape))

        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, env.action_space.n),
        )

    def forward(self, x):
        return self.net(x)

    def act(self, obs):
        obs_t = torch.tensor(obs, dtype=torch.float32)
        q_values = self(obs_t.unsqueeze(0))
        max_q_index = torch.argmax(q_values, dim=1)[0]
        action = max_q_index.detach().item()

        return action


env = gym.make("CartPole-v1")

memory = deque(maxlen=BUFFER_SIZE)
rew_buffer = deque([0.0], maxlen=100)

episode_reward = 0.0

online_net = Network(env)
target_net = Network(env)

target_net.load_state_dict(online_net.state_dict())

optimizer = torch.optim.Adam(online_net.parameters(), lr=5e-4)

# Initialize Replay Buffer
obs, _ = env.reset()
for _ in range(MIN_REPLAY_SIZE):
    action = env.action_space.sample()
    new_obs, rew, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    transition = (obs, action, rew, done, new_obs)
    memory.append(transition)
    obs = new_obs

    if done:
        obs, _ = env.reset()

# Training Loop
obs, _ = env.reset()
for step in itertools.count():
    epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])

    if random.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = online_net.act(obs)

    new_obs, rew, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    transition = (obs, action, rew, done, new_obs)
    memory.append(transition)
    obs = new_obs

    episode_reward += rew

    if done:
        obs, _ = env.reset()
        rew_buffer.append(episode_reward)
        episode_reward = 0.0

    # Watch it play after solving (checking 100 episodes played and mean reward >= 195)
    if len(rew_buffer) == 100 and np.mean(rew_buffer) >= 475.0:
        print(f"\nSolved at step {step}! Launching visual demo...")
        
        # Create dedicated rendering environment
        render_env = gym.make("CartPole-v1", render_mode="human")
        test_obs, _ = render_env.reset()
        
        while True:
            action = online_net.act(test_obs)
            test_obs, _, term, trunc, _ = render_env.step(action)
            
            if term or trunc:
                test_obs, _ = render_env.reset()

    # Gradient step - Sample from memory (replay buffer)
    transitions = random.sample(memory, BATCH_SIZE)

    obses = np.asarray([t[0] for t in transitions])
    actions = np.asarray([t[1] for t in transitions])
    rews = np.asarray([t[2] for t in transitions])
    dones = np.asarray([t[3] for t in transitions])
    new_obses = np.asarray([t[4] for t in transitions])

    obses_t = torch.as_tensor(obses, dtype=torch.float32)
    actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
    rews_t = torch.as_tensor(rews, dtype=torch.float32).unsqueeze(-1)
    dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1)
    new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32)

    # Compute targets
    target_q_values = target_net(new_obses_t)
    max_q_target_values = target_q_values.max(dim=1, keepdim=True)[0]
    targets = rews_t + GAMMA * (1 - dones_t) * max_q_target_values

    # Compute loss
    q_values = online_net(obses_t)
    action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)
    loss = nn.functional.smooth_l1_loss(action_q_values, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(online_net.state_dict())

    if step % 1000 == 0:
        print(f"Step {step} | Avg Rew: {np.mean(rew_buffer):.2f}")