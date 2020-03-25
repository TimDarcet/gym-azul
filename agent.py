import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import gym
from gym_azul.envs.azul_env import AzulEnv


def preprocess(state):
    """
    converts a state to a tensor
    """
    return torch.Tensor(state)


def postprocess(action, player_id, n_repos):
    """
    converts an action (int in [0, 6 * 5 * (N_REPOS + 1)]) to a dictionnary
    """
    rep, col, row = n_repos + 1, 5, 6
    i, j, k = action % rep, (action // rep) % col, (action // (rep * col)) % row
    env_action = {'player_id': player_id, 'take': {'repo': i, 'color': j}, 'put': k}
    return env_action



class Actor(nn.Module):
    """
    Policy network: given a state, produces a distribution on actions
    """
    def __init__(self, state_dim, action_dim, hidden_dim=30):
        super().__init__()
        self.lin1 = nn.Linear(state_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, input, logit=False):
        temp = self.lin1(input)
        score = self.lin2(F.relu(temp))
        if not logit:
            return F.softmax(score)
        return score

class Critic(nn.Module):
    """
    Value network: computes the expected value of a state
    """

    def __init__(self, state_dim, hidden_dim=30):
        super().__init__()
        self.lin1 = nn.Linear(state_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, 1)

    def forward(self, input):
        temp = self.lin1(input.view(-1))
        score = self.lin2(F.relu(temp))
        return score


class RandomAgent:
    """
    An agent playing randomly
    """

    def play(self, state, env, player_id):
        action = env.sample_action()
        new_state, _, done, _ = env.step(action)
        return new_state, done

    def update(self):
        pass


class Agent:
    """
    An agent learning with A2C
    """
    def __init__(self, state_dim, action_dim, hidden_dim, actor_optim, critic_optim, actor_lr= 1e-2, critic_lr=1e-3, gamma=0.9, nb_channels=1):
        # initializes networks
        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.critic = Critic(state_dim, hidden_dim)

        # if nb_channels > 1, the agent plays for several players (or in several games) and the histories must be tracked parallely
        self.nb_channels = nb_channels

        # creates empty reward, value and log odds records
        self.rewards = [[] for _ in range(self.nb_channels)]
        self.values = [[] for _ in range(self.nb_channels)]
        self.logodds = [[] for _ in range(self.nb_channels)]

        self.n_actions = action_dim
        self.gamma = gamma

        # initializes optimizers
        self.actor_optim = actor_optim(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = critic_optim(self.critic.parameters(), lr=critic_lr)

        self.stats = {'victories': 0, 'games': 0}

    def play(self, state, env, player_id):
        channel_id = player_id  # only considers the case where the agent plays for different players, no different games !!
        state = preprocess(state)  # converts into tensor
        distr = self.actor(state)
        action = np.random.choice(list(range(self.n_actions)), p=distr.detach().numpy())
        new_state, reward, done, _ = env.step(postprocess(action, player_id, env.n_repos))
        self.rewards[channel_id].append(reward)
        self.values[channel_id].append(self.critic(state))
        self.logodds[channel_id].append(torch.log(distr[action]))
        return new_state, done

    def update(self):
        """
        assumes that the rewards, values and loggodds sequences are contiguous in each channel
        """
        for channel in range(self.nb_channels):
            Q_values = torch.zeros(len(self.values[channel]))  # computes approximated Q-values with rewards
            Q_values[-1] = self.values[channel][-1].detach()
            for i in range(0, len(self.values[channel]) - 1, -1):
                Q_values[i] = self.rewards[channel][i] + self.gamma * Q_values[i+1]
            advantages = Q_values - torch.stack(self.values[channel])

            actor_loss = - (torch.stack(self.logodds[channel]) * advantages.detach()).mean()  # advantages is static for the actor
            critic_loss = 0.5 * advantages.pow(2).mean()

            self.actor_optim.zero_grad()
            self.critic_optim.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optim.step()
            self.critic_optim.step()
        self.reset()

    def reset(self):
        self.rewards = [[] for _ in range(self.nb_channels)]
        self.values = [[] for _ in range(self.nb_channels)]
        self.logodds = [[] for _ in range(self.nb_channels)]



