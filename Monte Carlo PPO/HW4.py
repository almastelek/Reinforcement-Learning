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
    if figure and len(episode_returns) % 100 == 0: 
        plt.savefig(figure)



Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'terminated'))
class ReplayMemory(object):
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
        # baseline (V_\phi) network
        self.fc4 = nn.Linear(dim, 128)
        self.fc5 = nn.Linear(128, 128)
        self.fc6 = nn.Linear(128, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=LR)
        self.M = M
        
    def forward(self, x):
        """
        The Q_theta in DQN or pi_theta in PPO
        """
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.fc3(x)
        if self.algorithm == "PPO": 
            x = nn.Softmax(dim=1)(x)
        return x

    def forward_baseline(self, x): 
        """
        The V_phi network in the pseudocode of PPO
        """
        x = self.fc4(x)
        x = nn.ReLU()(x)
        x = self.fc5(x)
        x = nn.ReLU()(x)
        x = self.fc6(x)
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
    

    def update_batch(self, batch_state, batch_action, batch_reward, batch_next_state, batch_terminated, batch_action_prob, memory=None, target_net=None):
        batch_action_prob = batch_action_prob.detach()
        if self.algorithm == "DQN": 
            ## This part is done in HW3
            for state, action, reward, next_state, terminated in zip(batch_state, batch_action, batch_reward, batch_next_state, batch_terminated):
                memory.push(state, action, reward, next_state, terminated)
            for _ in range(self.M):  
                states, actions, rewards, next_states, terminateds = zip(*memory.sample(MINIBATCH_SIZE))
                batch_state = torch.stack(states)
                batch_action = torch.stack(actions)
                batch_reward = torch.stack(rewards)
                batch_next_state = torch.stack(next_states)
                batch_terminated = torch.stack(terminateds)

                state_action_values = self.get_state_action_values(batch_state, batch_action)
                with torch.no_grad():
                    max_value_actions = target_net.get_max_value_actions(batch_next_state)
                    next_state_values = target_net.get_state_action_values(batch_next_state, max_value_actions) * (1 - batch_terminated)
                expected_state_action_values = batch_reward + GAMMA * next_state_values

                loss = F.mse_loss(state_action_values, expected_state_action_values) 
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                for target_param, policy_param in zip(target_net.parameters(), self.parameters()):
                    target_param.data.copy_(TAU * policy_param.data + (1.0 - TAU) * target_param.data)

        elif self.algorithm == "PPO":

            ## TODO 1: Calculate \hat{G}, \hat{A}, \hat{V} in the pseudocode for the N samples in the batch
            ##         Note that \hat{\pi} in the pseudocode is simply batch_action_prob 
            

            for _ in range(self.M):
                pass
                ## TODO 2: Sample a minibatch of size MINIBATCH_SIZE from the batch of size N
                ## TODO 3: Calculate the policy (\pi_\theta) update objective over the minibatch
                ## TODO 4: Calculate the value (V_\phi) update objective over the minibatch
                ## TODO 5: Update policy and value

               

    def act(self, x, iteration):
        q_values = self(x)
        if self.algorithm == "RAND":
            return 1 / self.n_actions * torch.ones_like(q_values)

        elif self.algorithm == "DQN": 
            eps = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * iteration / EPS_DECAY)
            max_index = q_values.argmax(dim=1)
            batchsize = x.shape[0]
            prob = torch.zeros_like(q_values)
            prob[torch.arange(batchsize), max_index] = 1.0
            prob = (1-eps) * prob + eps / self.n_actions * torch.ones_like(q_values)  
            return prob
    
        elif self.algorithm == "PPO":  
            return self(x)  
    




def train(args): 
    set_seed(seed=args.seed) 
    num_episodes = 2000


    policy_net = MDPModel(state_dim, n_actions, args)     # the online network in DQN or the policy network in PPO
    if args.algorithm == "DQN": 
        target_net = MDPModel(state_dim, n_actions, args)
        target_net.load_state_dict(policy_net.state_dict())   
        memory = ReplayMemory(1000000)
    else: 
        target_net = None
        memory = None

    
    batch_state = []
    batch_action = []
    batch_reward = []
    batch_next_state = []
    batch_terminated = []
    batch_action_prob = []
    episode_returns = []
    t = 0

    for iteration in range(num_episodes):
        current_episode_return = 0
        state, _ = env.reset(seed=9876543210*args.seed+iteration*(246+args.seed))
        terminated = truncated = 0

        while not (terminated or truncated):
            state = torch.tensor(state).unsqueeze(0)
            
            action_prob_all = policy_net.act(state, iteration)
            action = torch.multinomial(action_prob_all, num_samples=1).item()
            action_prob = action_prob_all[0,action]
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
            batch_action_prob.append(action_prob)


            if (t + 1) % N == 0: 
                batch_state = torch.cat(batch_state, dim=0)
                batch_action = torch.stack(batch_action, dim=0)
                batch_reward = torch.stack(batch_reward, dim=0)
                batch_next_state = torch.stack(batch_next_state, dim=0)
                batch_terminated = torch.stack(batch_terminated, dim=0)
                batch_action_prob = torch.stack(batch_action_prob, dim=0)
                policy_net.update_batch(
                    batch_state = batch_state, 
                    batch_action = batch_action, 
                    batch_reward = batch_reward, 
                    batch_next_state = batch_next_state, 
                    batch_terminated = batch_terminated, 
                    batch_action_prob = batch_action_prob,
                    memory = memory, 
                    target_net = target_net
                )
                batch_state = []
                batch_action = []
                batch_reward = []
                batch_next_state = []
                batch_terminated = []
                batch_action_prob = []
            
                
        
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
    parser.add_argument("--algorithm", type=str, required=True, default="DQN", help="DQN or PPO")
    parser.add_argument("--figure", type=str, required=False, default="CartPole_output.png")
    args = parser.parse_args()

    ##### Shared parameters between DQN and PPO #####
    MINIBATCH_SIZE = 128      # the B in the pseudocode
    GAMMA = 0.99
    LR = 1e-4
    if args.algorithm == "DQN" or args.algorithm == "RAND": 
       N = 4
       M = 4
    elif args.algorithm == "PPO": 
       N = 1024                # the N in the pseudocode
       M = 32                  # the M in the pseudocode


    ##### DQN-specific parameters #####
    EPS_START = 0.9
    EPS_END = 0.01
    EPS_DECAY = 500           # decay rate of epsilon (the larger the slower decay)
    TAU = 0.01                # the update rate of the target network 


    ##### PPO-specific parameters #####
    EPS = 0.1
    

    train(args)

