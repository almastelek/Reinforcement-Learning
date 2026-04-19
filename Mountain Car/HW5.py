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
import os
from datetime import datetime
from mountaincar_bonus import BonusStatistics, TabularExplorationBonus, RandomNetworkDistillation, plot_bonus_heatmap



def create_model_directory(args):
    """Create timestamped model directory with algorithm and exploration info"""
    if args.nosaving:
        return "."  # Use current directory for figures
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = f"models/{args.algorithm}_{args.exploration}_{args.seed}_{timestamp}"
        os.makedirs(model_dir, exist_ok=True)
        return model_dir


def detect_algorithm_from_filename(model_path):
    """Detect algorithm type from model filename"""
    filename = model_path.split('/')[-1]  # Get just the filename
    
    if 'PPO' in filename:
        return 'PPO'
    elif 'DQN' in filename:
        return 'DQN'  # DQN and DDQN have same architecture
    else:
        raise ValueError(f"Cannot detect algorithm from filename: {filename}. Expected filename to contain 'PPO', 'DQN'.")


def set_seed(seed):
    random.seed(seed)                   # Python random module
    np.random.seed(seed)                 # NumPy
    torch.manual_seed(seed)              # PyTorch CPU
    torch.cuda.manual_seed(seed)         # PyTorch GPU (if used)
    torch.cuda.manual_seed_all(seed)     # PyTorch multi-GPU
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.benchmark = False  # Disable cuDNN auto-tuner to prevent non-determinism


def plot_progress(episode_returns, model_dir=None, additional_returns=None):
    running_avg = [np.mean(episode_returns[max(0, i - 100 + 1):i + 1]) for i in range(len(episode_returns))]
    max_avg = max(running_avg)
    
    plt.figure(1)
    plt.clf()
    plt.title(f'Training... (Max Running Avg: {max_avg:.2f})')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    
    x = range(0, len(episode_returns))
    
    # Plot all per-episode curves first
    plt.plot(x, episode_returns, label='Learner Episode Returns')
    if additional_returns is not None:
        additional_running_avg = [np.mean(additional_returns[max(0, i - 100 + 1):i + 1]) for i in range(len(additional_returns))]
        plt.plot(x, additional_returns, label=f'Expert Episode Returns')
    
    # Plot all running average curves after (so they appear on top)
    plt.plot(x, running_avg, label=f'Learner Running Average')
    if additional_returns is not None:
        plt.plot(x, additional_running_avg, label=f'Expert Running Average')
    
    plt.legend()
    plt.pause(0.001)  
    
    if len(episode_returns) % 100 == 0:
        if model_dir:
            output_name = f"{model_dir}/return_curve.png"
        else:
            output_name = 'return_curve.png'
        plt.savefig(output_name)
        print(f"Return curve saved to: {output_name}")


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
        
        # Policy network
        self.fc1 = nn.Linear(dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)
        
        # Value network
        self.value_fc1 = nn.Linear(dim, 128)
        self.value_fc2 = nn.Linear(128, 128)
        self.value_fc3 = nn.Linear(128, 1)
        
        # Bonus value network  
        self.bonus_fc1 = nn.Linear(dim, 128)
        self.bonus_fc2 = nn.Linear(128, 128)
        self.bonus_fc3 = nn.Linear(128, 1)
        
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
        if self.algorithm == "PPO" or self.algorithm == "BC": 
            x = nn.Softmax(dim=1)(x)
        return x

    def forward_baseline(self, x): 
        """The V_phi network in the pseudocode of PPO (kept for compatibility)"""
        return self.forward_value(x)
    
    def forward_value(self, x):
        """Value network"""
        x = F.relu(self.value_fc1(x))
        x = F.relu(self.value_fc2(x))
        x = self.value_fc3(x)
        return x
    
    def forward_bonus_value(self, x):
        """Bonus value network"""
        x = F.relu(self.bonus_fc1(x))
        x = F.relu(self.bonus_fc2(x))
        x = self.bonus_fc3(x)
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
    

    def update_batch(self, batch_state, batch_action, batch_reward, batch_next_state, batch_terminated, batch_action_prob, memory=None, target_net=None, exploration_bonus=None):
        batch_action_prob = batch_action_prob.detach()

        if self.algorithm == "BC":
            ########################################################
            #  TODO-3.1:  push batch_state, batch_action to memory  
            ########################################################
            for state, action in zip(batch_state, batch_action):
                memory.push(state, action, torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0))
            
            for _ in range(self.M):
                pass
                ################################################################
                #  TODO-3.2:  sample a batch of states, actions from the memory
                #  TODO-3.3:  train the policy network for one step
                ################################################################
                states, actions, _, _, _ = zip(*memory.sample(MINIBATCH_SIZE))
                bc_states = torch.stack(states)
                bc_actions = torch.stack(actions)

                loss = F.cross_entropy(self(bc_states), bc_actions)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
               

        elif self.algorithm == "DQN":
            ####################################################### 
            #  NOTE:  DQN was done in Homework 3
            #  the only change below is that the reward is augmented with bonus. 
            #  In fact, the implementation below is DDQN. 
            #  Nothing to do here. 
            #######################################################
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
            
                augmented_rewards = batch_reward.clone()
                if exploration_bonus is not None:
                    for i, state in enumerate(batch_state):
                        bonus = exploration_bonus.get_bonus(state)
                        augmented_rewards[i] += bonus
            
                with torch.no_grad():
                    next_actions = self.get_max_value_actions(batch_next_state)
                    next_state_values = target_net.get_state_action_values(batch_next_state, next_actions) * (1 - batch_terminated)
                    expected_state_action_values = augmented_rewards + GAMMA * next_state_values

                loss = F.mse_loss(state_action_values, expected_state_action_values) 
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                for target_param, policy_param in zip(target_net.parameters(), self.parameters()):
                    target_param.data.copy_(TAU * policy_param.data + (1.0 - TAU) * target_param.data)

        elif self.algorithm == "PPO":
            #########################################################
            #  NOTE: A variant of PPO was done in Homework 4.  
            #  Here is a slightly more complicated version (see pseudocode). 
            #  The reward is augmented with bonus. 
            #  Nothing to do here. 
            #########################################################
            bonuses = torch.zeros_like(batch_reward)
            if exploration_bonus is not None:
                with torch.no_grad():
                    bonuses_list = []
                    for i, state in enumerate(batch_state):
                        bonus = exploration_bonus.get_bonus(state.cpu().numpy())
                        bonuses_list.append(bonus)
                    bonuses = torch.tensor(bonuses_list, dtype=torch.float32)
            
            with torch.no_grad():
                values = self.forward_value(batch_state).squeeze(-1)
                bonus_values = self.forward_bonus_value(batch_state).squeeze(-1)
                next_values = self.forward_value(batch_next_state).squeeze(-1)
                bonus_next_values = self.forward_bonus_value(batch_next_state).squeeze(-1)

            advantages = torch.zeros_like(batch_reward)
            bonus_advantages = torch.zeros_like(bonuses)
            gae = 0
            bonus_gae = 0
            
            for t in reversed(range(len(batch_reward))):
                # GAE
                delta = batch_reward[t] + GAMMA * next_values[t] * (1.0 - batch_terminated[t]) - values[t]
                gae = delta + GAMMA * LAMBDA * (1.0 - batch_terminated[t]) * gae
                advantages[t] = gae
                
                # Bonus GAE
                bonus_delta = bonuses[t] + GAMMA * bonus_next_values[t] * (1.0 - batch_terminated[t]) - bonus_values[t]
                bonus_gae = bonus_delta + GAMMA * LAMBDA * (1.0 - batch_terminated[t]) * bonus_gae
                bonus_advantages[t] = bonus_gae
            
            total_advantages = advantages + bonus_advantages
            
            returns = advantages + values.detach()
            bonus_returns = bonus_advantages + bonus_values.detach()
            
            for _ in range(self.M):
                indices = torch.randperm(len(batch_state))
                mb_indices = indices[0:MINIBATCH_SIZE]
                mb_states = batch_state[mb_indices]
                mb_actions = batch_action[mb_indices]
                mb_advantages = total_advantages[mb_indices]
                mb_returns = returns[mb_indices]
                mb_bonus_returns = bonus_returns[mb_indices]
                mb_old_log_probs = batch_action_prob[mb_indices]
                    
                new_probs_all = self(mb_states)
                new_action_prob = new_probs_all.gather(1, mb_actions.unsqueeze(1))

                ratios = new_action_prob / (mb_old_log_probs.unsqueeze(1))
                clipped_ratios = torch.clamp(ratios, min=1-EPS, max=1+EPS)
                
                # Policy losses    
                policy_loss = -torch.mean(torch.min(ratios * mb_advantages.unsqueeze(1), 
                                                  clipped_ratios * mb_advantages.unsqueeze(1)))
                    
                neg_entropy = torch.mean(torch.sum(new_probs_all * torch.log(new_probs_all + 1e-8), dim=1))    
                # Value losses
                value_pred = self.forward_value(mb_states).squeeze(-1)
                bonus_value_pred = self.forward_bonus_value(mb_states).squeeze(-1)
                value_loss = F.mse_loss(value_pred, mb_returns)
                bonus_value_loss = F.mse_loss(bonus_value_pred, mb_bonus_returns)
                    
                total_loss = (policy_loss + 
                             VF_COEF * value_loss + 
                             BONUS_VF_COEF * bonus_value_loss + 
                             BETA * neg_entropy)
                    
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()


    def act(self, x, iteration):
        q_values = self(x)
        if self.algorithm == "DQN":   
            eps = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * iteration / EPS_DECAY)
            max_index = q_values.argmax(dim=1)
            batchsize = x.shape[0]
            prob = torch.zeros_like(q_values)
            prob[torch.arange(batchsize), max_index] = 1.0
            prob = (1-eps) * prob + eps / self.n_actions * torch.ones_like(q_values)  
            return prob
    
        elif self.algorithm == "PPO":  
            eps = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * iteration / EPS_DECAY)
            policy_probs = self(x)
            uniform_probs = torch.ones_like(policy_probs) / self.n_actions
            mixed_probs = (1 - eps) * policy_probs + eps * uniform_probs
            return mixed_probs
            
        elif self.algorithm == "BC":
            return self(x)  
    

     
def imitate(args, n_actions, state_dim):  
    model_dir = create_model_directory(args)
    
    # Load expert model (user needs to specify expert_model_path)
    if not hasattr(args, 'expert_model_path') or args.expert_model_path is None:
        raise ValueError("For BC training, please specify --expert_model_path with the path to the trained expert model")
    
    
    # Detect expert algorithm from filename
    expert_algorithm = detect_algorithm_from_filename(args.expert_model_path)
    print(f"Detected expert algorithm: {expert_algorithm}")
    
    expert_args = argparse.Namespace(**vars(args))
    expert_args.algorithm = expert_algorithm
    expert_net = MDPModel(state_dim, n_actions, expert_args)
    expert_net.load_state_dict(torch.load(args.expert_model_path))
    expert_net.eval()
    
    # Initialize BC policy
    policy_net = MDPModel(state_dim, n_actions, args)
    memory = ReplayMemory(1000000)
    
    env = gym.make('MountainCar-v0')
    num_episodes = 1000
    expert_returns = []
    learner_returns = []
    
    batch_state = []
    batch_action = []
    batch_terminated = []
    t = 0
    
    for iteration in range(num_episodes): 
        # Expert rollout
        expert_return = 0
        state, _ = env.reset(seed=9876543210*args.seed+iteration*(246+args.seed))
        terminated = truncated = False
        
        while not (terminated or truncated):
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action_prob_all = expert_net(state)  # Direct forward pass for expert
            action = torch.multinomial(action_prob_all, num_samples=1).item()
            action_prob = action_prob_all[0,action]
            next_state, reward, terminated, truncated, _ = env.step(action)
            expert_return += reward

            action = torch.as_tensor(action)
            next_state = torch.as_tensor(next_state, dtype=torch.float32)
            terminated = torch.as_tensor(terminated).int()
       
            batch_state.append(state)
            batch_action.append(action)
            batch_terminated.append(terminated)
            
            if (t + 1) % N == 0:
                batch_state = torch.cat(batch_state, dim=0)
                batch_action = torch.stack(batch_action, dim=0)
                batch_terminated = torch.stack(batch_terminated, dim=0)

                policy_net.update_batch(
                    batch_state = batch_state, 
                    batch_action = batch_action, 
                    batch_reward = torch.zeros(len(batch_state)),  # dummy values
                    batch_next_state = batch_state,  # dummy values
                    batch_terminated = batch_terminated, 
                    batch_action_prob = torch.ones(len(batch_state)),  # dummy values
                    memory = memory 
                )

                batch_state = []
                batch_action = []
                batch_terminated = []

            if terminated or truncated: 
                expert_returns.append(expert_return)
                print(f'Expert Episode {iteration}, return: {expert_return}')
                
                # Evaluate learner performance
                learner_return = evaluate_learner(policy_net, env, args.seed, iteration)
                learner_returns.append(learner_return)
                print(f'Learner Episode {iteration}, return: {learner_return}')
                
                # Save BC model every 100 episodes
                if (iteration + 1) % 100 == 0 and not args.nosaving:
                    torch.save(policy_net.state_dict(), f"{model_dir}/model_BC_ep{iteration + 1}.pth")
                    print(f"BC model saved at episode {iteration + 1} in {model_dir}")
                
                plot_progress(learner_returns, model_dir=model_dir, additional_returns=expert_returns)
            else:
                state = next_state
            
            t = t+1
    

def evaluate_learner(policy_net, env, seed, iteration):
    """Evaluate learner policy for one episode"""
    policy_net.eval()
    with torch.no_grad():
        state, _ = env.reset(seed=seed*456+iteration)
        terminated = truncated = False
        total_return = 0
        
        while not (terminated or truncated):
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action_prob_all = policy_net.act(state, 0)  # No exploration
            action = torch.multinomial(action_prob_all, num_samples=1).item()
            state, reward, terminated, truncated, _ = env.step(action)
            total_return += reward
    
    policy_net.train()
    return total_return


def train(args, n_actions, state_dim): 
    set_seed(seed=args.seed) 
    num_episodes = 6000
    
    model_dir = create_model_directory(args)

    policy_net = MDPModel(state_dim, n_actions, args)     # the online network in DQN or the policy network in PPO
    if args.algorithm == "DQN" or args.algorithm == "DDQN": 
        target_net = MDPModel(state_dim, n_actions, args)
        target_net.load_state_dict(policy_net.state_dict())   
        memory = ReplayMemory(1000000)
    else: 
        target_net = None
        memory = None

    exploration_bonus = None
    env_temp = gym.make('MountainCar-v0')
    state_bounds = [[env_temp.observation_space.low[i], env_temp.observation_space.high[i]] for i in range(len(env_temp.observation_space.low))] #[[-1.2, 0.6], [-0.07, 0.07]] in mountain car
    env_temp.close()
    bonus_stats = BonusStatistics(num_bins=20, state_bounds=state_bounds)  # Always create for plotting visitation
    
    if args.exploration == "tabular":
        exploration_bonus = TabularExplorationBonus(state_dim=state_dim, num_bins=20, bonus_scale=0.5, state_bounds=state_bounds)
    elif args.exploration == "rnd":
        exploration_bonus = RandomNetworkDistillation(state_dim=state_dim, hidden_dim=64, bonus_scale=2, lr=1e-3, state_bounds=state_bounds)

    
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
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            
            action_prob_all = policy_net.act(state, iteration)
            action = torch.multinomial(action_prob_all, num_samples=1).item()
            action_prob = action_prob_all[0,action]
            next_state, reward, terminated, truncated, _ = env.step(action)
            current_episode_return += reward


            action = torch.as_tensor(action)
            reward = torch.as_tensor(reward, dtype=torch.float32)
            next_state = torch.as_tensor(next_state, dtype=torch.float32)
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
                    target_net = target_net,
                    exploration_bonus = exploration_bonus
                )
                
                                
                ###############################################################################
                # NOTE:  Update exploration bonus and statistics after training on this batch
                ###############################################################################
                for state in batch_state:
                    if exploration_bonus is not None:
                        bonus_value = exploration_bonus.train_and_get_bonus(state)
                    else:
                        bonus_value = 0
                    
                    bonus_stats.update(state, bonus_value)
                batch_state = []
                batch_action = []
                batch_reward = []
                batch_next_state = []
                batch_terminated = []
                batch_action_prob = []
                            
        
            if terminated or truncated:
                episode_returns.append(current_episode_return)
                print('Episode {},  score: {}'.format(iteration, current_episode_return), flush=True)
                
                # Visualize bonus heatmap every 20 episodes
                if (iteration + 1) % 20 == 0:
                    plot_bonus_heatmap(bonus_stats, iteration + 1, args, model_dir)
                
                # Save model every 100 episodes
                if (iteration + 1) % 100 == 0 and not args.nosaving:
                    torch.save(policy_net.state_dict(), f"{model_dir}/model_{args.algorithm}_ep{iteration + 1}.pth")
                    print(f"Model saved at episode {iteration + 1} in {model_dir}")
                
                plot_progress(episode_returns, model_dir=model_dir)    
                    
            else: 
                state = next_state

            t = t+1
    
    # Save the trained model for BC
    if not args.nosaving:
        torch.save(policy_net.state_dict(), f"{model_dir}/expert_model_{args.algorithm}.pth")
        print(f"Expert model saved as {model_dir}/expert_model_{args.algorithm}.pth")


if __name__ == "__main__": 
    env = gym.make('MountainCar-v0')
    n_actions = env.action_space.n
    state_dim = env.observation_space.shape[0]

    parser = argparse.ArgumentParser(description="Script with input parameters")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--algorithm", type=str, required=True, default="DQN", help="DQN or PPO or BC")
    parser.add_argument("--exploration", type=str, choices=["none", "tabular", "rnd"], default="none", 
                       help="Exploration strategy: none (no bonus), tabular (discretized state bonus), rnd (Random Network Distillation)")
    parser.add_argument("--expert_model_path", type=str, required=False, default=None, help="Path to expert model for BC training")
    parser.add_argument("--nosaving", action="store_true", help="Skip model saving and use current directory for figures")
    args = parser.parse_args()

    ##### Shared parameters between DQN and PPO #####
    MINIBATCH_SIZE = 128      # the B in the pseudocode
    GAMMA = 0.99              # Discount factor for rewards and bonuses
    EPS_START = 0.9
    EPS_END = 0.001
    EPS_DECAY = 500           # decay rate of epsilon (the larger the slower decay)

    if args.algorithm == "DQN": 
       LR = 1e-4
       N = 4                 
       M = 1
    elif args.algorithm == "PPO": 
       LR = 1e-3              # Learning rate
       N = 2048               # Steps per update
       M = 32                 # Epochs per update
    elif args.algorithm == "BC":
       LR = 1e-4              # Learning rate for BC
       N = 4                  # Update frequency
       M = 4                  # Number of BC updates per expert episode



    ##### DQN-specific parameters #####
    TAU = 0.01                # the update rate of the target network 


    ##### PPO-specific parameters #####
    LAMBDA = 0.95             # GAE lambda
    EPS = 0.2                 # PPO clipping range
    BETA = 0.01               # Entropy coefficient
    VF_COEF = 0.5             # Value function coefficient
    BONUS_VF_COEF = 1.0       # Bonus value function coefficient
    

    if args.algorithm == "BC":
        imitate(args, n_actions, state_dim)
    else:
        train(args, n_actions, state_dim)

    

