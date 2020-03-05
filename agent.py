import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import gym
from gym_azul.envs.azul_env import AzulEnv


def preprocess(state):
    """
    TODO: converts a state to a tensor
    """
    pass


def postprocess(action):
    """
    converts an action (integer) to an env.action
    """




class Actor(nn.Module):
    """
    Policy network: given a state, produces a distribution on actions
    """
    def __init__(self, state_dim, action_dim, hidden_dim=30):
        super().__init__()
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

    def __init__(self, state_dim, hidden_dim=30):
        super().__init__()
        self.lin1 = nn.Linear(state_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, 1)

    def forward(self, input):
        temp = self.lin1(input)
        score = self.lin2(F.relu(temp))
        return score



class Agent:
    """
    An agent that can play against others
    """
    def __init__(self, state_dim, action_dim, hidden_dim, actor_optim, critic_optim, gamma):
        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.critic = Critic(state_dim, hidden_dim)
        self.rewards = []
        self.values = []
        self.logodds = []
        self.n_actions = action_dim
        self.gamma = gamma
        self.actor_optim = actor_optim
        self.critic_optim = critic_optim

    def play(self, state, env):
        state = preprocess(state)  # converts into tensor
        distr = self.actor(state)
        action = np.random.choice(list(range(self.n_actions)), p=distr)
        new_state, reward, done, _ = env.step(postprocess(action))
        self.rewards.append(reward)
        self.values.append(self.critic(state))
        self.logodds.append(torch.log(distr))
        return done

    def update(self):
        """
        assumes that the rewards, values and loggodds sequences are continuous
        """
        Q_values = torch.like(self.values)  # computes approximated Q-values with rewards
        Q_values[-1] = self.values[-1].detach()
        for i in range(0, len(self.values) - 1, -1):
            Q_values[i] = self.rewards[i] + self.gamma * Q_values[i+1]
        advantages = Q_values - self.values

        actor_loss = - self.logodds * advantages.detach()
        critic_loss = 0.5 * advantages.pow(2).mean()
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optim.step()
        self.critic_optim.step()
        self.reset()

    def reset(self):
        self.rewards = []
        self.values = []
        self.logodds = []



