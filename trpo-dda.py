"""
TRPO-DDA Clean Implementation 

This version:
- Keeps full TRPO pipeline
- Removes proprietary adaptive control details
- Provides simplified, reproducible surrogate mechanisms

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import copy
import math
from torch.distributions import Normal


# =========================================================
# 1. Initialization Utilities
# =========================================================
def init_net_weights(m):
    """Simple uniform initialization for stability."""
    if isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight, -0.01, 0.01)
        nn.init.constant_(m.bias, 0.0)


# =========================================================
# 2. Value Network (Critic)
# =========================================================
class ValueNetContinuous(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        self.apply(init_net_weights)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


# =========================================================
# 3. Policy Network (Gaussian Actor)
# =========================================================
class PolicyNetContinuous(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound
        self.apply(init_net_weights)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        mu = torch.tanh(self.mu(x)) * self.action_bound
        std = F.softplus(self.log_std(x)) + 1e-6
        return mu, std


# =========================================================
# 4. Advantage Estimation (GAE)
# =========================================================
def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().cpu().numpy()
    adv = 0
    out = []
    for d in td_delta[::-1]:
        adv = gamma * lmbda * adv + d
        out.append(adv)
    out.reverse()
    return torch.tensor(out, dtype=torch.float32)


# =========================================================
# 5. TRPO-DDA 
# =========================================================
class TRPOContinuousClean:
    """
    Clean TRPO version with simplified adaptive components.

    NOTE:
    All advanced mechanisms in TRPO-DDA (risk-aware entropy,
    covariance-driven shaping, and dynamic trust-region logic)
    are replaced with stable placeholders.
    """

    def __init__(self,
                 hidden_dim,
                 state_space,
                 action_space,
                 gamma=0.99,
                 lmbda=0.97,
                 kl_constraint=5e-5,
                 alpha=0.5,
                 critic_lr=5e-4,
                 device="cpu"):

        state_dim = state_space.shape[0]
        action_dim = action_space.shape[0]
        action_bound = action_space.high[0]

        self.device = device
        self.gamma = gamma
        self.lmbda = lmbda
        self.kl_constraint = kl_constraint
        self.alpha = alpha

        # Networks
        self.actor = PolicyNetContinuous(
            state_dim, hidden_dim, action_dim, action_bound
        ).to(device)

        self.critic = ValueNetContinuous(
            state_dim, hidden_dim
        ).to(device)

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr
        )

        # =====================================================
        # Simplified entropy control (placeholder for DDA)
        # =====================================================
        self.log_beta = torch.nn.Parameter(torch.tensor(-11.5, device=device))
        self.beta_optimizer = torch.optim.Adam([self.log_beta], lr=3e-4)

        self.entropy_ema = None
        self.ema_decay = 0.9

        # Trust region adaptation (simplified)
        self.kl_min = 3e-5
        self.kl_max = 1e-4

    # =====================================================
    # Action sampling
    # =====================================================
  
    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float32).to(self.device)
        mu, std = self.actor(state)
        dist = Normal(mu, std)
        action = dist.sample()
        return action.detach().cpu().numpy().flatten().tolist()

    # =====================================================
    # TRPO Core Utilities
    # =====================================================
  
    #Compute the product of a Hessian matrix and a vector
    def hessian_vector_product(self, states, old_dist, vector, damping=0.01):
        mu, std = self.actor(states)
        new_dist = Normal(mu, std)

        kl = torch.distributions.kl.kl_divergence(old_dist, new_dist).mean()
        grads = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)
        flat = torch.cat([g.view(-1) for g in grads])

        kl_v = torch.dot(flat, vector)
        hvp = torch.autograd.grad(kl_v, self.actor.parameters())
        hvp = torch.cat([g.contiguous().view(-1) for g in hvp])

        return hvp + damping * vector
      
    #Solve a set of linear equations with conjugate descent method
    def conjugate_gradient(self, g, states, old_dist):
        x = torch.zeros_like(g)
        r = g.clone()
        p = g.clone()

        for _ in range(10):
            Hp = self.hessian_vector_product(states, old_dist, p)
            alpha = torch.dot(r, r) / (torch.dot(p, Hp) + 1e-8)
            x += alpha * p
            r_new = r - alpha * Hp
            beta = torch.dot(r_new, r_new) / (torch.dot(r, r) + 1e-8)
            p = r_new + beta * p
            r = r_new
        return x

    # =====================================================
    # Simplified surrogate objective
    # =====================================================
    #The objective function with entropy regularization after the transformation of the original objective
    def surrogate(self, states, actions, adv, old_logp):
        mu, std = self.actor(states)
        dist = Normal(mu, std)

        logp = dist.log_prob(actions).sum(dim=-1)
        ratio = torch.exp(logp - old_logp)

        entropy = dist.entropy().sum(dim=-1).mean()

        beta = self.log_beta.exp().detach()

        return torch.mean(ratio * adv) + beta * entropy

    # =====================================================
    # Trust region (SIMPLIFIED)
    # =====================================================
    def update_kl(self, accept_ratio):
        """
        Simplified version:
        only proportional adaptation (no multi-state logic)
        """
        if accept_ratio < 0.9:
            self.kl_constraint *= 0.8
        elif accept_ratio > 1.1:
            self.kl_constraint *= 1.05

        self.kl_constraint = np.clip(self.kl_constraint,
                                     self.kl_min,
                                     self.kl_max)

    # =====================================================
    # Line search
    # =====================================================
    def line_search(self, states, actions, adv, old_logp, old_dist,
                    step, g):

        old_params = torch.nn.utils.parameters_to_vector(
            self.actor.parameters()
        ).detach()

        old_obj = self.surrogate(states, actions, adv, old_logp).detach()

        g_dot_step = torch.dot(g, step)

        for i in range(15):
            coef = self.alpha ** i
            new_params = old_params + coef * step

            new_actor = copy.deepcopy(self.actor)
            torch.nn.utils.vector_to_parameters(new_params, new_actor.parameters())

            mu, std = new_actor(states)
            new_dist = Normal(mu, std)

            kl = torch.distributions.kl.kl_divergence(old_dist, new_dist).mean()

            new_obj = self.surrogate(states, actions, adv, old_logp)

            if (new_obj > old_obj) and (kl < self.kl_constraint):
                return new_params, coef

        return old_params, 0.0

    # =====================================================
    # Policy update
    # =====================================================
    def policy_update(self, states, actions, old_dist, old_logp, adv):

        loss = self.surrogate(states, actions, adv, old_logp)
        grads = torch.autograd.grad(loss, self.actor.parameters())
        g = torch.cat([x.view(-1) for x in grads]).detach()

        step_dir = self.conjugate_gradient(g, states, old_dist)

        hess = self.hessian_vector_product(states, old_dist, step_dir)
        scale = torch.sqrt(2 * self.kl_constraint /
                           (torch.dot(step_dir, hess) + 1e-8))

        full_step = step_dir * scale

        new_params, coef = self.line_search(
            states, actions, adv, old_logp, old_dist, full_step, g
        )

        torch.nn.utils.vector_to_parameters(new_params, self.actor.parameters())

        # simplified accept ratio proxy
        self.update_kl(accept_ratio=1.0)

    # =====================================================
    # Placeholder for entropy update (DDA removed)
    # =====================================================
    def update_entropy(self, entropy):
    # =====================================================
    # Critic update
    # =====================================================
    def update_critic(self, states, target):
        loss = F.mse_loss(self.critic(states), target)
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

    # =====================================================
    # Update entry
    # =====================================================
    def update(self, batch):
        states = torch.tensor(batch['states'], dtype=torch.float32).to(self.device)
        actions = torch.tensor(batch['actions'], dtype=torch.float32).to(self.device)
        rewards = torch.tensor(batch['rewards'], dtype=torch.float32).view(-1,1).to(self.device)
        next_states = torch.tensor(batch['next_states'], dtype=torch.float32).to(self.device)
        done = torch.tensor(batch['done'], dtype=torch.float32).view(-1,1).to(self.device)

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - done)
        td_delta = td_target - self.critic(states)

        adv = compute_advantage(self.gamma, self.lmbda, td_delta)

        mu, std = self.actor(states)
        old_dist = Normal(mu.detach(), std.detach())

        old_logp = old_dist.log_prob(actions).sum(dim=-1)

        self.update_critic(states, td_target.detach())

        self.policy_update(states, actions, old_dist, old_logp, adv)

        return {}

# =========================================================
# 6. Simple Training Loop (Single Seed Version)
# =========================================================

import matplotlib.pyplot as plt
from collections import deque


def moving_average(data, window=50):
    if len(data) < window:
        return np.array(data)
    return np.convolve(data, np.ones(window)/window, mode='valid')


def train_single_seed(env_name="Swimmer-v5",
                      hidden_dim=128,
                      total_episodes=6000,
                      seed=0,
                      device="cpu"):

    # -------------------------
    # Environment
    # -------------------------
    env = gym.make(env_name)
    env.reset(seed=seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = TRPOContinuousClean(
        hidden_dim=hidden_dim,
        state_space=env.observation_space,
        action_space=env.action_space,
        device=device
    )

    # -------------------------
    # Logging
    # -------------------------
    reward_list = []
    avg_reward_list = []

    reward_window = deque(maxlen=50)

    # -------------------------
    # Training loop
    # -------------------------
    for episode in range(total_episodes):

        state, _ = env.reset()
        done = False

        episode_reward = 0
        transition = {
            "states": [],
            "actions": [],
            "rewards": [],
            "next_states": [],
            "done": []
        }

        while not done:

            action = agent.take_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            transition["states"].append(state)
            transition["actions"].append(action)
            transition["rewards"].append(reward)
            transition["next_states"].append(next_state)
            transition["done"].append(done)

            state = next_state
            episode_reward += reward

        # -------------------------
        # Update agent
        # -------------------------
        metrics = agent.update(transition)

        # -------------------------
        # Logging reward
        # -------------------------
        reward_list.append(episode_reward)
        reward_window.append(episode_reward)

        avg_reward = np.mean(reward_window)
        avg_reward_list.append(avg_reward)

        # -------------------------
        # Print progress
        # -------------------------
        if episode % 10 == 0:
            print(f"[Episode {episode}] "
                  f"Reward: {episode_reward:.2f} | "
                  f"Avg(50): {avg_reward:.2f} | "
                  f"KL: {agent.kl_constraint:.2e}")

    env.close()

    return reward_list, avg_reward_list

# =========================================================
# 7. Plot Results
# =========================================================

def plot_rewards(reward_list, avg_reward_list, save_path="reward_curve.png"):

    plt.figure(figsize=(10,5))

    plt.plot(reward_list, label="Episode Reward", alpha=0.3)
    plt.plot(avg_reward_list, label="Moving Avg (50)", linewidth=2)

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Curve (Single Seed)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.show()

if __name__ == "__main__":

    rewards, avg_rewards = train_single_seed(
        env_name="HalfCheetah-v5",
        hidden_dim=128,
        total_episodes=1000,
        seed=0,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    plot_rewards(rewards, avg_rewards)

# =========================================================
# 8. Multi-Seed Training Framework
# =========================================================

def run_multi_seed(
        env_name="HalfCheetah-v5",
        seeds=[seed1, seed2, seed3, seed4, seed5],
        total_episodes=6000,
        hidden_dim=128,
        device="cpu"):

    all_rewards = []
    all_avg_rewards = []

    for seed in seeds:

        print(f"\n==============================")
        print(f"Running seed {seed}")
        print(f"==============================\n")

        rewards, avg_rewards = train_single_seed(
            env_name=env_name,
            hidden_dim=hidden_dim,
            total_episodes=total_episodes,
            seed=seed,
            device=device
        )

        all_rewards.append(rewards)
        all_avg_rewards.append(avg_rewards)

    # 转 numpy
    all_rewards = np.array(all_rewards)        # [S, T]
    all_avg_rewards = np.array(all_avg_rewards)

    return all_rewards, all_avg_rewards

# =========================================================
# 9. Multi-Seed Plot (Mean ± Std)
# =========================================================

def plot_multi_seed(all_rewards, window=50, save_path="multi_seed_curve.png"):

    all_rewards = np.array(all_rewards)
    T = all_rewards.shape[1]

    mean = np.mean(all_rewards, axis=0)
    std = np.std(all_rewards, axis=0)

    # smoothing
    mean_smooth = np.convolve(mean, np.ones(window)/window, mode='valid')
    std_smooth = np.convolve(std, np.ones(window)/window, mode='valid')

    x = np.arange(len(mean_smooth))

    plt.figure(figsize=(10,5))

    plt.plot(x, mean_smooth, label="Mean Return")
    plt.fill_between(x,
                     mean_smooth - std_smooth,
                     mean_smooth + std_smooth,
                     alpha=0.3)

    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Multi-Seed Performance (Mean ± Std)")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.show()
  
