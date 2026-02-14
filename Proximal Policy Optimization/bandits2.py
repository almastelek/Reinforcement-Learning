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
        elif args.algorithm == "PPO":
            self.b = args.b
            self.alpha = args.alpha
            self.baseline_type = args.baseline
            self.tau = args.tau

        self.fc1 = nn.Linear(dim, 32)
        self.fc2 = nn.Linear(32, n_actions)     

        # NOTE: baseline network for PPO
        self.fc3 = nn.Linear(dim, 32)
        self.fc4 = nn.Linear(32, 1)
   
        self.optimizer = optim.Adam(self.parameters(), lr=5e-4) 
        # The number of data collection steps within a single t in 1,2,...,T
        self.N = 16
        # The number of gradient steps within a single t in 1,2,...,T
        self.M = 10

        # self.alpha = 0.1


    def forward(self, x):
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        # NOTE: For PPO, we add a softmax layer to the end of the network
        if self.algorithm == "PPO":
            x = nn.Softmax(dim=1)(x)
        return x

    # NOTE: this function can calculate baseline value b(x) for PPO
    def forward_baseline(self, x): 
        x = self.fc3(x)
        x = nn.ReLU()(x)
        x = self.fc4(x)
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
            ## This was done in HW1
            eps = float(self.eps)
            prob = torch.full_like(gap, eps / self.n_actions)
            best = predicted_reward_all.argmax(dim=1)             # (B,)
            prob[torch.arange(batchsize), best] += (1.0 - eps)
            return prob

        elif self.algorithm == "BE":
            ## This was done in HW1  
            lam = float(self.ld)
            return F.softmax(lam * predicted_reward_all, dim=1)


        elif self.algorithm == "PPO": 
            return self(x)
        

    def update_batch(self, batch_x, batch_action, batch_reward, batch_action_prob):
        """
        Args:
            batch_x: a batch of contexts
            batch_action: the batch of actions chosen in those contexts
            batch_reward: the batch of rewards observed
            batch_action_prob: the batch of probabilities for only those chosen actions
        """
        if self.algorithm == "PPO":
            reward_prediction = self.forward_baseline(batch_x)    # this is b_\phi(x) in the pseudocode

            # calculate batch_reward_estimator, which is the a batch of "r_{t,n} - b_{t,n}" in the pseudocode
            # advantage term: was action better than expected?
            if self.baseline_type == "static": 
                 batch_reward_estimator = batch_reward - self.b
            elif self.baseline_type == "dynamic": 
                 batch_reward_estimator = batch_reward - reward_prediction - self.b

            # "detach" the following variables so the gradient won't propagate to them
            batch_reward_estimator = batch_reward_estimator.detach()
            batch_action_prob = batch_action_prob.detach()

            for _ in range(self.M):

                # TODO: Compute the training loss for PPO and perform gradient descent 
                # That is, coding up Eq.(1) and (3) in the pseudocode and call optimizer.step() once
                # batch_x: contexts

                # Get the new action probabilities from the current policy
                new_action_probs_all = self(batch_x)  # shape: [batch_size, n_actions]
                # Gather the probabilities of the actions that were actually taken
                new_action_prob = torch.gather(new_action_probs_all, 1, batch_action)  # shape: [batch_size, 1]

                importance_ratio = new_action_prob / batch_action_prob  # shape: [batch_size, 1]

                # KL divergence between old and new policy (over the batch)
                kl_est = importance_ratio - 1.0 - torch.log(importance_ratio)

                # PPO loss: maximize core_policy_gradient, minimize KL divergence
                loss = -(importance_ratio * batch_reward_estimator - self.alpha * kl_est).mean()

                if self.baseline_type == "dynamic":
                    reward_prediction = self.forward_baseline(batch_x)
                    baseline_loss = F.mse_loss(reward_prediction, batch_reward)  # MSE loss for baseline network
                    loss += self.tau * baseline_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        else:   # regression oracle for value based approaches (BE, EG), done in HW1
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
    elif args.algorithm == "PPO":
        print("b=", args.b, "  alpha=", args.alpha, end="")
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
            recent = accuracy_values[max(0,t-500): t+1] # updated to be t+1 to include the current reward in the recent average (otherwise RuntimeWarning: invalid value encountered in scalar divide)
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
        choices=["Rand", "EG", "BE", "Greedy", "PPO"],
        help="Algorithm to run",
    )
    parser.add_argument("--baseline", type=str, default="static", choices=["static", "dynamic"])
    parser.add_argument("--tau", type=float, default=1.0)

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
    # elif args.algorithm == "PPO": 
    #     b_list = [0.3, 0.15, 0.0, -0.15, -0.3]#[2.0, 1.5, 1.0, 0.5, 0.0, -0.5]
    #     alpha_list = [0.1]
    #     plt.figure()
    #     for b in b_list:
    #         for alpha in alpha_list:
    #             args.b = b
    #             args.alpha = alpha
    #             curve = train(args, return_curve=True) 
    #             plt.plot(np.arange(len(curve)), curve, label=f"b={b}, alpha={alpha}")
    #     plt.title("Running Average Reward (PPO)")
    #     plt.xlabel("Time Step")
    #     plt.ylabel("Average Reward")
    #     plt.legend()
    #     plt.savefig("running_avg_PPO.png", dpi=150, bbox_inches="tight")
    #     plt.show() 
    
    # for KL regularization sweep
    elif args.algorithm == "PPO":
        # Use best baseline setting from section 1.1
        args.b = 1.0                   # best static baseline
        args.baseline = "static"       # or dynamic if you choose that
        alpha_list = [0, 0.1, 0.2, 0.5, 1.0]

        plt.figure()
        for alpha in alpha_list:
            args.alpha = alpha
            curve = train(args, return_curve=True)
            plt.plot(np.arange(len(curve)), curve, label=f"alpha={alpha}")
        plt.title("Running Average Reward (PPO) - Alpha Sweep")
        plt.xlabel("Time Step")
        plt.ylabel("Average Reward")
        plt.legend()
        plt.savefig("running_avg_PPO_alpha_sweep.png", dpi=150, bbox_inches="tight")
        plt.show()
    else:
        train(args)
