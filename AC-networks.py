import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueNetContinuous(nn.Module):
    """
    State-value network.
    """

    def __init__(
        self,
        state_dim,
        hidden_dim
    ):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.out(x)

class PolicyNetContinuous(nn.Module):
    """
    Gaussian policy network used in TRPO-DDA.
    """

    def __init__(
        self,
        state_dim,
        hidden_dim,
        action_dim,
        action_bound
    ):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.fc_mu = nn.Linear(hidden_dim, action_dim)
        self.fc_std = nn.Linear(hidden_dim, action_dim)

        self.action_bound = action_bound

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        mu = self.action_bound * torch.tanh(
            self.fc_mu(x)
        )

        std = F.softplus(
            self.fc_std(x)
        )

        return mu, std
