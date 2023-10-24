import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from gymnasium.wrappers import GrayScaleObservation
from gymnasium.wrappers import FrameStack
from gymnasium.wrappers import ResizeObservation

import time


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
    
    
### env
task = "Pong"
env = GrayScaleObservation(gym.make("ALE/Pong-v5", render_mode="rgb_array"), keep_dim=True)
env = ResizeObservation(env, 84)
env = FrameStack(env, 4)
print(env.observation_space)

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
print(is_ipython)
if is_ipython:
    from IPython import display

# plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))




class DQN(nn.Module):
    def __init__(self, in_channels=3, n_actions=14):
        """
        Initialize Deep Q Network

        Args:
            in_channels (int): number of input channels
            n_actions (int): number of outputs
            w * h = 84
        """
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        # self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # self.bn3 = nn.BatchNorm2d(64)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.head = nn.Linear(512, n_actions)
        
    def forward(self, x):
        x = x.float() / 255
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.head(x)




# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 256
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 100000
TAU = 0.005
LR = 1e-5
SEED = 525

torch.manual_seed(seed=SEED)
env.action_space.seed(seed=SEED)
random.seed(SEED)

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
n_frames = env.frames.maxlen
print(f"n_actions: {n_actions}, \tn_frames: {n_frames}")

# init policy_net and target_net, target_net parameter == policy_net parameter

# policy_net = torch.load('/home/yen/code/yen/RL/checkpoint/DQN_Pong_800_5.pt') # TODO
policy_net = DQN(n_frames, n_actions).to(device)
target_net = DQN(n_frames, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(50000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


episode_durations = []


# def plot_durations(show_result=False):
#     plt.figure(1)
#     durations_t = torch.tensor(episode_durations, dtype=torch.float)
#     if show_result:
#         plt.title('Result')
#     else:
#         plt.clf()
#         plt.title('Training...')
#     plt.xlabel('Episode')
#     plt.ylabel('Duration')
#     plt.plot(durations_t.numpy())
#     # Take 100 episode averages and plot them too
#     if len(durations_t) >= 100:
#         means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
#         means = torch.cat((torch.zeros(99), means))
#         plt.plot(means.numpy())

#     plt.pause(0.001)  # pause a bit so that plots are updated
#     if is_ipython:
#         if not show_result:
#             display.display(plt.gcf())
#             display.clear_output(wait=True)
#         else:
#             display.display(plt.gcf())
            
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))
    # print(transitions)
    # print(zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    
if torch.cuda.is_available():
    num_episodes = 1000
    # num_episodes = 1
else:
    num_episodes = 50




all_total_rewards = []
for i_episode in range(1, 1+num_episodes):
    # Initialize the environment and get it's state
    state, info = env.reset(seed=SEED+i_episode)
    # w, h, c = state.shape
    # env.render()
    state = torch.tensor(np.array(state), dtype=torch.float32, device=device).view(1, 4, 84, -1)   ### (batch, channel, width, height)
    # state = pad_resize(state, 84)   
    # print(state.shape, "\n")
    
    
    total_reward = 0

    # print(state.shape)
    start = time.time()
    
    env_times_all = 0
    model_times_all = 0
    
    for t in count():
        env_start = time.time()
        action = select_action(state) 
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            # next_state = pad_resize(observation, 84)  
            next_state = torch.tensor(np.array(observation), dtype=torch.float32, device=device).view(1, 4, 84, -1)
            
            
        total_reward += reward

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        env_end = time.time()
        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        model_end = time.time()
        
        env_times_all += env_end - env_start
        model_times_all += model_end - env_end
        
        
        if done:
            end = time.time()
            
            episode_durations.append(t + 1)
            all_total_rewards.append(total_reward)
            # plot_durations()
            print(f"episode: {i_episode} \tdurations: {t}\t\trewards: {total_reward.item()} \ttimes: {round(end-start, 2)}")
            print(f"times: {round(end-start, 2)}, env: {round(env_times_all, 2), round(env_times_all/(end-start), 2)}, model:{round(model_times_all, 2), round(model_times_all/(end-start), 2)}")
            break
        
    if i_episode % 50 == 0:
        torch.save(target_net, f"./checkpoint/e/DQN_{task}_{i_episode}_{int(sum(all_total_rewards[-10:])/10)}.pt")
        torch.save(target_net.state_dict(), f"./checkpoint/e/DQN_{task}_{i_episode}_{int(sum(all_total_rewards[-10:])/10)}_state_dict.pt")
        print(f"{i_episode},  save model, reward:{int(sum(all_total_rewards[-10:])/10)}")

print('Complete')
