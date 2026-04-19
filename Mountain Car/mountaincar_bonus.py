import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class BonusStatistics:
    def __init__(self, num_bins, state_bounds):
        self.num_bins = num_bins
        self.visit_counts = {}
        self.bin_bonuses = {}  # bin -> list of bonus values
        self.state_bounds = state_bounds
        
    def _discretize_state(self, state):
        """Convert continuous state to discrete bins"""
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy().flatten()
        elif not isinstance(state, np.ndarray):
            state = np.array(state).flatten()
            
        discrete_state = []
        for i in range(len(state)):
            min_val, max_val = self.state_bounds[i]
            bin_size = (max_val - min_val) / self.num_bins
            bin_idx = int((state[i] - min_val) / bin_size)
            bin_idx = max(0, min(self.num_bins - 1, bin_idx))  # Clamp to valid range
            discrete_state.append(bin_idx)
        
        return tuple(discrete_state)
    
    def update(self, state, bonus_value):
        """Update statistics for a given state and its bonus value"""
        discrete_state = self._discretize_state(state) 
        self.visit_counts[discrete_state] = self.visit_counts.get(discrete_state, 0) + 1
        
        if discrete_state not in self.bin_bonuses:
            self.bin_bonuses[discrete_state] = []
        self.bin_bonuses[discrete_state].append(bonus_value)
    
    
    def get_2d_bonus_map(self):
        bonus_map = np.zeros((self.num_bins, self.num_bins))
        for (pos_bin, vel_bin), bonuses in self.bin_bonuses.items():
            if len(bonuses) > 0:
                bonus_map[vel_bin, pos_bin] = np.mean(bonuses)
                
        return bonus_map
    
    def get_2d_visitation_map(self):
        visitation_map = np.full((self.num_bins, self.num_bins), -20.0)
        total_visits = sum(self.visit_counts.values()) if self.visit_counts else 0
        
        if total_visits > 0:
            for (pos_bin, vel_bin), count in self.visit_counts.items():
                percentage = (count / total_visits) * 100
                log_percentage = np.log(percentage)
                visitation_map[vel_bin, pos_bin] = log_percentage
                
        return visitation_map


class TabularExplorationBonus:
    def __init__(self, state_dim, num_bins, bonus_scale, state_bounds):
        self.state_dim = state_dim
        self.num_bins = num_bins
        self.bonus_scale = bonus_scale
        self.visit_counts = {}
        self.state_bounds = np.array(state_bounds)
        
    def _discretize_state(self, state):
        """Convert continuous state to discrete bins"""
        # Convert to numpy array regardless of input type
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy().flatten()
        else:
            state = np.array(state).flatten()
            
        # Vectorized discretization
        min_vals = self.state_bounds[:, 0]
        max_vals = self.state_bounds[:, 1]
        bin_sizes = (max_vals - min_vals) / self.num_bins
        bin_indices = ((state - min_vals) / bin_sizes).astype(int)
        bin_indices = np.clip(bin_indices, 0, self.num_bins - 1)
        
        return tuple(bin_indices)
    
    def get_bonus(self, state, update_counts=False):
        discrete_state = self._discretize_state(state)

        ########################################
        #  TODO-1:
        #  1. Calculate the bonus as self.bonus_scale / sqrt( self.visit_counts[discrete_state] )
        #     If discrete_state has not been visited before, initialize self.visit_counts[discrete_state] as 1.
        #  2. If update_counts == True, add 1 to self.visit_counts[discrete_state].
        #  3. Return the bonus.
        ########################################
        bonus = self.bonus_scale / np.sqrt(self.visit_counts.get(discrete_state, 1))
        if update_counts:
            self.visit_counts[discrete_state] = self.visit_counts.get(discrete_state, 0) + 1

        return bonus
    
    def train_and_get_bonus(self, state):
        return self.get_bonus(state, update_counts=True)


class RandomNetworkDistillation:
    def __init__(self, state_dim, hidden_dim, bonus_scale, lr, state_bounds):
        self.state_dim = state_dim
        self.bonus_scale = bonus_scale
        self.state_bounds = np.array(state_bounds)
        
        # Target network (fixed, randomly initialized)
        self.target_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Predictor network (trainable)
        self.predictor_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Freeze target network
        for param in self.target_net.parameters():
            param.requires_grad = False
            
        self.optimizer = optim.Adam(self.predictor_net.parameters(), lr=lr)
        
        
    def _normalize_state(self, state):
        """Normalize state based on known state space bounds"""
        if isinstance(state, torch.Tensor):
            state_tensor = state.flatten().unsqueeze(0)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).flatten().unsqueeze(0)
            
        min_vals = torch.tensor(self.state_bounds[:, 0], dtype=torch.float32)  # [-1.2, -0.07]
        max_vals = torch.tensor(self.state_bounds[:, 1], dtype=torch.float32)  # [0.6, 0.07]
        
        # Normalize to [-1, 1] range
        normalized = 2 * (state_tensor - min_vals) / (max_vals - min_vals) - 1
        return normalized
        
        
    def get_bonus(self, state, train=False):
        """Get exploration bonus based on prediction error, optionally training the network"""
        state_norm = self._normalize_state(state)   # Normalize the state
       
        ###############################################
        #  TODO-2:
        #  1. Calculate the bonus as the two-norm squared distance bewteen self.target_net(state_norm)
        #     and self.predictor_net(state_norm).
        #  2. If train == True, perform one gradient update to minimize the
        #     distance between self.target_net(state_norm) and self.predictor_net(state_norm).
        #     Notice that the target_net should be fixed --- only train the predictor_net.
        #  3. Set the bonus as self.bonus_scale * (the two-norm square distance above)
        #  4. Return the bonus
        ###############################################
        with torch.no_grad():
            target_output = self.target_net(state_norm)
        predictor_output = self.predictor_net(state_norm)
        
        distance_squared = ((predictor_output - target_output) ** 2).sum()
        bonus = self.bonus_scale * distance_squared
        
        if train:
            loss = F.mse_loss(predictor_output, target_output)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return bonus.item()

    
    def train_and_get_bonus(self, state):
        return self.get_bonus(state, train=True)


def plot_bonus_heatmap(bonus_stats, episode, args, model_dir=None):
    """Plot 2D heatmap of bonus values and visitation percentages"""
    if bonus_stats is None:
        return
    
    # Get state bounds from bonus_stats
    state_bounds = bonus_stats.state_bounds
    # Create extent for imshow: [x_min, x_max, y_min, y_max] = [pos_min, pos_max, vel_min, vel_max]
    extent = [state_bounds[0][0], state_bounds[0][1], state_bounds[1][0], state_bounds[1][1]]
    
    plt.figure(2, figsize=(12, 5))
    plt.clf()
    
    # Subplot 1: Bonus values (empty for "none" exploration)
    plt.subplot(1, 2, 1)
    if args.exploration != "none":
        bonus_map = bonus_stats.get_2d_bonus_map()
        im1 = plt.imshow(bonus_map, cmap='viridis', aspect='auto', origin='lower',
                        extent=extent)
        plt.colorbar(im1, label='Average Bonus Value')
        plt.title(f'Bonus Values - Episode {episode}')
    else:
        # Empty plot for "none" exploration
        plt.imshow(np.zeros((20, 20)), cmap='viridis', aspect='auto', origin='lower',
                  extent=extent)
        plt.colorbar(label='Average Bonus Value')
        plt.title(f'Bonus Values - Episode {episode} (no exploration)')
    
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Visitation percentages
    plt.subplot(1, 2, 2)
    visitation_map = bonus_stats.get_2d_visitation_map()
    im2 = plt.imshow(visitation_map, cmap='hot', aspect='auto', origin='lower',
                    extent=extent)
    plt.colorbar(im2, label='Log(Visitation Percentage)')
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.title(f'State Visitation - Episode {episode}')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.pause(0.001)
    
    # Save the figure every 100 episodes
    if episode % 100 == 0:
        if model_dir:
            bonus_name = f"{model_dir}/bonus_heatmap.png"
        else:
            bonus_name = 'bonus.png'
        plt.savefig(bonus_name)
        print(f"Bonus and visitation heatmaps saved to: {bonus_name}")
