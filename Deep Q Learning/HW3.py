## This code is for lunar lander
import gymnasium as gym
import math
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse
import imageio



def set_seed(seed):
    random.seed(seed)                   # Python random module
    np.random.seed(seed)                 # NumPy
    torch.manual_seed(seed)              # PyTorch CPU
    torch.cuda.manual_seed(seed)         # PyTorch GPU (if used)
    torch.cuda.manual_seed_all(seed)     # PyTorch multi-GPU
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.benchmark = False  # Disable cuDNN auto-tuner to prevent non-determinism


def plot_progress(episode_returns, figure=None):
    running_avg = [np.mean(episode_returns[max(0, i - 100 + 1):i + 1]) for i in range(len(episode_returns))]
    max_avg = max(running_avg)
    plt.figure(1)
    plt.clf()
    plt.title(f'Training... (Max Running Avg: {max_avg:.2f})')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    x = range(0, len(episode_returns))
    plt.plot(x, episode_returns, label='Episode Returns')
    plt.plot(x, running_avg, label=f'Running Average')
    plt.pause(0.001)  
    if len(episode_returns) % 100 == 0: 
        plt.savefig(figure)


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'terminated'))
class ReplayBuffer(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args): 
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        if batch_size <= len(self.memory):
            return random.sample(self.memory, batch_size)
        else: 
            return random.choices(self.memory, k=batch_size)
        
    def __len__(self):
        return len(self.memory)


class MDPModel(nn.Module):
    def __init__(self, dim, n_actions, args):
        super(MDPModel, self).__init__()
        self.dim = dim
        self.n_actions = n_actions
        self.algorithm = args.algorithm
        self.fc1 = nn.Linear(dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=LR)
        self.M = M
        
    def forward(self, x):
        """
        The Q_theta in DQN
        """
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.fc3(x)
        return x

        
    def get_state_action_values(self, batch_state, batch_action):   
        """
        return Q[s,a]
        """
        q_values = self(batch_state)
        row_index = torch.arange(0, batch_state.shape[0])
        selected_actions_q_values = q_values[row_index, batch_action]
        return selected_actions_q_values
    
    def get_state_values(self, batch_state): 
        """
        return max_a Q[s,a]
        """
        q_values = self(batch_state)
        max_q_values = q_values.max(dim=1).values
        return max_q_values
    
    def get_max_value_actions(self, batch_state): 
        """
        return argmax_a Q[s,a]
        """
        q_values = self(batch_state)
        max_q_actions = q_values.max(dim=1).indices
        return max_q_actions
    

    def update_batch(self, batch_state, batch_action, batch_reward, batch_next_state, batch_terminated, buffer, target_net):
        # TODO 1:  insert (s,a,r,s',t) in the input batch to the buffer
        for i in range(len(batch_state)):
            buffer.push(batch_state[i], batch_action[i], batch_reward[i], batch_next_state[i], batch_terminated[i])
        
        for m in range(self.M):
            # TODO 2.1: sample a new batch of tuples (s,a,r,s',t) of size MINIBATCH_SIZE from the buffer
            samples = buffer.sample(MINIBATCH_SIZE)
            batch = Transition(*zip(*samples))
            sampled_state = torch.stack(batch.state)
            sampled_action = torch.stack(batch.action)
            sampled_reward = torch.stack(batch.reward)
            sampled_next_state = torch.stack(batch.next_state)
            sampled_terminated = torch.stack(batch.terminated)

            # TODO 2.2: calculate the regression loss 
            #           (self.get_state_action_values(s, a) - r - GAMMA * (1-t) * target_net.get_state_values(s'))**2
            #           for this new batch  
            #           and perform a gradient descent step
            #           Importantly, the gradient should not be taken on target_net

            # Double DQN
            # Vectorized Q(s, a) from main network
            q_pred = self.get_state_action_values(sampled_state, sampled_action)

            # Vectorized Q'(s') from target network
            with torch.no_grad():
                target_max_actions = self.get_max_value_actions(sampled_next_state)
                q_next = target_net.get_state_action_values(sampled_next_state, target_max_actions)
                target = sampled_reward + GAMMA * (1 - sampled_terminated) * q_next
            
            # MSE loss
            loss = F.mse_loss(q_pred, target)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Previous TODO 2.2 Single DQN
            # # Vectorized Q(s, a) from main network
            # q_pred = self.get_state_action_values(sampled_state, sampled_action)

            # # Vectorized Q'(s') from target network
            # with torch.no_grad():
            #     q_next = target_net.get_state_values(sampled_next_state)
            #     target = sampled_reward + GAMMA * (1 - sampled_terminated) * q_next
            
            # # MSE loss
            # loss = F.mse_loss(q_pred, target)
            # loss.backward()
            # self.optimizer.step()
            # self.optimizer.zero_grad()

            # TODO 2.3: set the target_net parameter as (1-TAU) * target_net parameter + TAU * main_net (self) parameter
            for target_param, main_param in zip(target_net.parameters(), self.parameters()):
                target_param.data.copy_((1 - TAU) * target_param.data + TAU * main_param.data)
            

    def act(self, x, iteration):
        q_values = self(x)
        if self.algorithm == "RAND":
            return 1 / self.n_actions * torch.ones_like(q_values)

        elif self.algorithm == "DQN" or self.algorithm == "DDQN":   
            eps = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * iteration / EPS_DECAY)
            max_index = q_values.argmax(dim=1)
            batchsize = x.shape[0]
            prob = torch.zeros_like(q_values)
            prob[torch.arange(batchsize), max_index] = 1.0
            prob = (1-eps) * prob + eps / self.n_actions * torch.ones_like(q_values)  
            return prob
    


def train(args): 
    set_seed(seed=args.seed) 
    num_episodes = 2000


    main_net = MDPModel(state_dim, n_actions, args)  
    target_net = MDPModel(state_dim, n_actions, args)
    target_net.load_state_dict(main_net.state_dict())   
    buffer = ReplayBuffer(1000000)

    
    batch_state = []
    batch_action = []
    batch_reward = []
    batch_next_state = []
    batch_terminated = []
    episode_returns = []
    t = 0

    for iteration in range(num_episodes):
        current_episode_return = 0
        state, _ = env.reset(seed=9876543210*args.seed+iteration*(246+args.seed))
        terminated = truncated = 0

        while not (terminated or truncated):
            state = torch.tensor(state).unsqueeze(0) 
            action_prob_all = main_net.act(state, iteration)
            action = torch.multinomial(action_prob_all, num_samples=1).item()
            next_state, reward, terminated, truncated, _ = env.step(action)
            current_episode_return += reward

            action = torch.as_tensor(action)
            reward = torch.as_tensor(reward)
            next_state = torch.as_tensor(next_state)
            terminated = torch.as_tensor(terminated).int()
        
            batch_state.append(state)
            batch_action.append(action)
            batch_reward.append(reward)
            batch_next_state.append(next_state)
            batch_terminated.append(terminated)


            if (t + 1) % N == 0: 
                batch_state = torch.cat(batch_state, dim=0)
                batch_action = torch.stack(batch_action, dim=0)
                batch_reward = torch.stack(batch_reward, dim=0)
                batch_next_state = torch.stack(batch_next_state, dim=0)
                batch_terminated = torch.stack(batch_terminated, dim=0)
                main_net.update_batch(
                    batch_state = batch_state, 
                    batch_action = batch_action, 
                    batch_reward = batch_reward, 
                    batch_next_state = batch_next_state, 
                    batch_terminated = batch_terminated, 
                    buffer = buffer, 
                    target_net = target_net
                )
                batch_state = []
                batch_action = []
                batch_reward = []
                batch_next_state = []
                batch_terminated = []
            
                
        
            if terminated or truncated:
                episode_returns.append(current_episode_return)
                print('Episode {},  score: {}'.format(iteration, current_episode_return), flush=True)
                plot_progress(episode_returns, figure=args.figure)    
                    
            else: 
                state = next_state

            t = t+1


if __name__ == "__main__": 
    env = gym.make('CartPole-v1')
    n_actions = env.action_space.n
    state_dim = env.observation_space.shape[0]

    parser = argparse.ArgumentParser(description="Script with input parameters")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--algorithm", type=str, required=True, default="DQN", help="DQN or DDQN")
    parser.add_argument("--figure", type=str, required=False, default=None)
    args = parser.parse_args()

    if args.figure == None: 
       args.figure = "HW3_output_" + args.algorithm + ".png"



    MINIBATCH_SIZE = 128      # the B in the pseudocode
    GAMMA = 0.99
    LR = 1e-4
    if args.algorithm == "DQN" or args.algorithm == "DDQN" or args.algorithm == "RAND": 
       N = 4
       M = 4

    EPS_START = 0.9
    EPS_END = 0.01
    EPS_DECAY = 500           # decay rate of epsilon (the larger the slower decay)
    TAU = 0.01                # the update rate of the target network 


    train(args)

