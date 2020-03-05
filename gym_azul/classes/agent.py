import torch.nn as nn
import torch
import torch.nn.functional as F
import gym
from gym_azul.envs.azul_env import AzulEnv


class Actor(nn.Module):
    """
    Policy network: given a state, produces a distribution on actions
    """
    def __init__(self, state_dim, action_dim, hidden_dim=30):
        self.lin1 = nn.Linear(state_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, input, logit=True):
        temp = self.lin1(input)
        score = self.lin2(F.relu(temp))
        if not logit:
            return F.softmax(score)
        return  score

class Critic(nn.Module):
    """
    Value network: computes the expected value of a state
    """

    def __init__(self, state_dim, hidden_dim=30, gamma=0.9):
        self.lin1 = nn.Linear(state_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, 1)
        self.gamma = gamma

    def forward(self, input):
        temp = self.lin1(input)
        score = self.lin2(F.relu(temp))
        return score

    def advantage(self, state, next_state, reward):
        return reward + self.gamma * self.forward(next_state) - self.forward(state)

