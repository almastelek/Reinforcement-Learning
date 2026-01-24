import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


# Load MNIST dataset
mnist = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())

X = mnist.data.numpy().reshape(-1, 784).astype(np.float32)
y = mnist.targets.numpy()

# Select 2500 samples per class
X_list, y_list = [], []
for digit in range(4):
    X_d, y_d = X[y == digit][:2500], y[y == digit][:2500]
    X_list.append(X_d)
    y_list.append(y_d)

# Concatenate the selected samples
X = np.vstack(X_list)
y = np.hstack(y_list)

print("X.shape (digit images)", X.shape)
print("y.shape (digit labels)", y.shape)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)
train_dataset = TensorDataset(X_tensor, y_tensor)
context_dim = X_tensor.shape[1]
n_actions = len(torch.unique(y_tensor))


class BanditModel(nn.Module):
    def __init__(self, dim, n_actions, args):
        super().__init__()
        self.dim = dim
        self.n_actions = n_actions
        self.algorithm = args.algorithm
        if args.algorithm == "EG": 
            self.eps = args.eps
        elif args.algorithm == "BE": 
            self.ld = args.ld

        self.fc1 = nn.Linear(dim, 32)
        self.fc2 = nn.Linear(32, n_actions)        
        self.optimizer = optim.Adam(self.parameters(), lr=5e-4) 
        # The number of data collection steps within a single t in 1,2,...,T
        self.N = 16
        # The number of gradient steps within a single t in 1,2,...,T
        self.M = 10

    def forward(self, x):
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x

    def act(self, x, time):
        predicted_reward_all = self(x)
        max_predicted_reward = torch.max(predicted_reward_all, dim=1, keepdim=True)[0]
        gap = predicted_reward_all - max_predicted_reward 
        batchsize = gap.shape[0] 

        if self.algorithm == "Rand":
            return 1 / self.n_actions * torch.ones_like(gap)  

        elif self.algorithm == "Greedy":
            max_index = gap.argmax(dim=1)
            prob = torch.zeros_like(gap)
            prob[torch.arange(batchsize), max_index] = 1
            return prob

        elif self.algorithm == "EG":  
            prob = self.eps / self.n_actions * torch.ones_like(gap)
            max_index = gap.argmax(dim=1)
            prob[torch.arange(batchsize), max_index] += 1 - self.eps
            return prob

        elif self.algorithm == "BE":
            exp_term = torch.exp(self.ld * gap)
            sum_exp = torch.sum(exp_term, dim=1, keepdim=True)
            prob = exp_term / sum_exp
            return prob

    def update_batch(self, batch_x, batch_action, batch_reward, batch_action_prob):
        """
        Args:
            batch_x: a batch of contexts
            batch_action: the batch of actions chosen in those contexts
            batch_reward: the batch of rewards observed
            batch_action_prob: the batch of probabilities for only those chosen actions
        """
        # regression oracle for value based approaches (BE, EG)
        for _ in range(self.M):
            all_predicted_rewards = self(batch_x)  
            predicted_rewards = all_predicted_rewards.gather(1, batch_action)
            loss = F.mse_loss(predicted_rewards, batch_reward)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


def train(args, n_seeds=5, return_curve=False):
    """
    Main training loop. 
    """
    T = y.shape[0]
    accuracy_values_avg = np.zeros((T,))
    accuracy_values_recent_avg = np.zeros((T,))

    print(f"##### Algorithm: {args.algorithm} with ", end="")
    if args.algorithm == "EG":
        print("eps =", args.eps, end="")
    elif args.algorithm == "BE":
        print("lambda =", args.ld, end="")
    print(" #####")



    for sd in range(n_seeds): 
        torch.manual_seed(sd)
        random.seed(sd)
        np.random.seed(sd)

        print("running for seed = ", sd)

        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        model = BanditModel(context_dim, n_actions, args)

        accuracy_values = np.zeros((T,))
        accuracy_values_recent = np.zeros((T,))
        t = 0
        batch_context = []
        batch_action = []
        batch_reward = []
        batch_action_prob = []
        
        
        for context, label in train_loader:
            switch_point = y.shape[0] // 2
            if t < switch_point: 
                reward_vector = 0.5 * F.one_hot(label, n_actions).float() 
            else:  
                reward_vector = 0.5 * F.one_hot(label, n_actions).float() + F.one_hot( (label+1)%n_actions, n_actions).float() 
            reward_vector += 0.5 * torch.randn_like(reward_vector)

                
            action_prob_all = model.act(context, t)
            
            action = torch.multinomial(action_prob_all, num_samples=1)
            reward = torch.gather(reward_vector, 1, action)
            action_prob = torch.gather(action_prob_all, 1, action)

            batch_context.append(context)
            batch_action.append(action)
            batch_reward.append(reward)
            batch_action_prob.append(action_prob)
                    
            if (t + 1) % model.N == 0:
                batch_context = torch.cat(batch_context, dim=0)
                batch_action = torch.cat(batch_action, dim=0)
                batch_reward = torch.cat(batch_reward, dim=0)
                batch_action_prob = torch.cat(batch_action_prob, dim=0)
                model.update_batch(
                    batch_context, batch_action, batch_reward, batch_action_prob
                )
                batch_context = []
                batch_action = []
                batch_reward = []
                batch_action_prob = []
                        

            # Common accuracy tracking
            accuracy_values[t] = reward
            recent = accuracy_values[max(0,t-500): t]
            accuracy_values_recent[t] = np.mean(recent)
            t += 1

        accuracy_values_avg += accuracy_values
        accuracy_values_recent_avg += accuracy_values_recent
    
    accuracy_values_avg /= n_seeds
    accuracy_values_recent_avg /= n_seeds

   
    print("average_reward_1st_phase =", np.mean(accuracy_values_avg[:switch_point]))
    print("average_reward_2nd_phase =", np.mean(accuracy_values_avg[switch_point:]))
    print("average_reward =", np.mean(accuracy_values_avg))

    if return_curve:
        return accuracy_values_recent_avg

    plt.figure()
    plt.plot(np.arange(T), accuracy_values_recent_avg)
    plt.title(f"Average Reward over Time ({args.algorithm})")
    plt.xlabel("Time Step")
    plt.ylabel("Average Reward")
    plt.savefig(f"running_avg_{args.algorithm}.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run bandit algorithms")
    parser.add_argument(
        "--algorithm",
        type=str,
        default="Rand",
        choices=["Rand", "EG", "BE", "Greedy"],
        help="Algorithm to run",
    )

    args = parser.parse_args()
    if args.algorithm == "EG":
        eps_list = [0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0]
        plt.figure()
        for eps in eps_list:
            args.eps = eps
            curve = train(args, return_curve=True)
            plt.plot(np.arange(len(curve)), curve, label=f"eps={eps}")
        plt.title("Running Average Reward (EG)")
        plt.xlabel("Time Step")
        plt.ylabel("Average Reward")
        plt.legend()
        plt.savefig("running_avg_EG.png", dpi=150, bbox_inches="tight")
        plt.show()
    elif args.algorithm == "BE":
        lam_list = [2, 5, 10, 20, 50]
        plt.figure()
        for lam in lam_list:
            args.ld = lam
            curve = train(args, return_curve=True) 
            plt.plot(np.arange(len(curve)), curve, label=f"lambda={lam}")
        plt.title("Running Average Reward (BE)")
        plt.xlabel("Time Step")
        plt.ylabel("Average Reward")
        plt.legend()
        plt.savefig("running_avg_BE.png", dpi=150, bbox_inches="tight")
        plt.show() 
    else:
        train(args)
