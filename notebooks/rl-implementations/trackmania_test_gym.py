import gymnasium as gym
import math
import time
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from pynput.keyboard import Controller

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils import get_active_window_process_id, get_process_id_by_name, check_if_tm2020_active
from trackmania_env import TrackmaniaEnv

import os
import sys
sys.path.insert(0, r'D:\study\bachelor\github\trackmania-ai\models')

while True:
    if check_if_tm2020_active():
        break

num_actions = 3
env = TrackmaniaEnv(num_actions=num_actions)

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
max_available_time = 15

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)

policy_best_model_path = r'models\rl_models\best_policy_dqn_model_epoch_1686.pt'
target_best_model_path = r'models\rl_models\best_target_dqn_model_epoch_1686.pt'

policy_net.load_state_dict(torch.load(policy_best_model_path))
target_net.load_state_dict(torch.load(target_best_model_path))

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


steps_done = 0

def select_action(state, is_validation=False):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if is_validation == True or sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


episode_durations = []


if torch.cuda.is_available():
    num_episodes = 2000
else:
    num_episodes = 50

keyboard = Controller()

episodes_numbers = []
episodes_rewards = []

best_reward = 0

for i_episode in range(num_episodes):
    # Initialize the environment and get it's state
    state, info = env.reset()
    time.sleep(3)
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    t = 0
    start_time = time.time()
    while True:
        # if not check_if_tm2020_active():        #     continue
        current_time = time.time()
        if current_time - start_time > max_available_time:
            print("New epoch has started")
            break
        
        action = select_action(state, 
                               is_validation=True)
        observation, reward, terminated, truncated, _ = env.step(action.item(), 
                                                                 reward_time=max_available_time - (current_time - start_time))
        print(reward)
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        if done:
            print("episode in done")
            episode_durations.append(t + 1)
            episodes_numbers.append(i_episode)
            episodes_rewards.append(reward)

            if best_reward < reward:
                best_reward = reward

            # plot_durations()
            break
        t += 1

print('Complete')
print(episodes_numbers)
print(episodes_rewards)
# plot_durations(show_result=True)
# plt.ioff()
# plt.show()
