import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import time
import random


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
            return F.softmax(score, dim=-1)
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


class Agent:
    """
    An abstract class for agents
    """
    def __init__(self):
        self.stats = {'victories': 0, 'games': 0}

    def play(self, state, env, player_id):
        # MUST BE OVERLOADED
        pass

    def update(self):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass

    def reset(self):
        pass

    def next_game(self, result):
        self.stats['victories'] += result
        self.stats['games'] += 1
        self.reset()  # to be sure

class RandomAgent(Agent):
    """
    An agent playing randomly
    """

    def __init__(self):
        super().__init__()

    def play(self, state, env, player_id):
        action = env.sample_action()
        new_state, _, done, _ = env.step(action)
        return new_state, done

class HumanAgent(Agent):
    """
    An agent that asks you to play
    """
    def __init__(self):
        super().__init__()

    def play(self, state, env, player_id):
        print('Your turn to play (player {})! Current state:'.format(player_id))
        played = False
        env.render()
        while not played:
            print('Your move (type "i j k" where i is a repo ID, j is a color ID and k is a queue ID):')
            try:
                repo, color, queue = input().split()
                repo, color, queue = int(repo), int(color), int(queue)
                assert 0 <= repo < env.n_repos and 0 <= color < 5 and 0 <= queue < 6
                action = {'player_id': player_id, 'take': {'repo': repo, 'color': color}, 'put': queue}
                new_state, _, done, _ = env.step(action)
            except (ValueError, AssertionError):
                print('Your action is invalid, please try again')
            else:
                played = True
        return new_state, done


class A2CAgents(Agent):
    """
    An agent learning with A2C
    """
    def __init__(self, state_dim, action_dim, hidden_dim, actor_optim, critic_optim, actor_lr= 1e-2, critic_lr=1e-3, gamma=0.9, nb_channels=1):
        super().__init__()

        # initializes networks
        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.critic = Critic(state_dim, hidden_dim)

        # if nb_channels > 1, the agent plays for several players (or in several games) and the histories must be tracked parallely
        self.nb_channels = nb_channels

        # creates empty reward, value and log odds records
        self.rewards = [[] for _ in range(self.nb_channels)]
        self.values = [[] for _ in range(self.nb_channels)]
        self.logodds = [[] for _ in range(self.nb_channels)]

        # some info
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma

        # initializes optimizers
        self.actor_optim = actor_optim(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = critic_optim(self.critic.parameters(), lr=critic_lr)

    def play(self, state, env, player_id):
        state = preprocess(state)  # converts into tensor
        distr = self.actor(state)
        action = np.random.choice(list(range(self.action_dim)), p=distr.detach().numpy())
        env.set_tolerant(True)  # so that the env samples a random action if the action is invalid
        new_state, reward, done, _ = env.step(postprocess(action, player_id, env.n_repos))
        env.set_tolerant(False)
        # records history for updating weights later
        if self.nb_channels > 1:
            channel_id = player_id  # only considers the case where the agent plays for different players, no different games !!
        else:
            channel_id = 0
        self.rewards[channel_id].append(reward)
        self.values[channel_id].append(self.critic(state))
        self.logodds[channel_id].append(torch.log(distr[action]))
        return new_state, done

    def update(self):
        """
        assumes that the rewards, values and loggodds sequences are contiguous in each channel
        """
        if not self.values[0]:
            return
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

    def next_game(self, result):
        self.stats['victories'] += result
        self.stats['games'] += 1
        self.reset()  # to be sure

    def save(self, path):
        torch.save({'shape': (self.state_dim, self.hidden_dim, self.action_dim),
                    'actor_weights': self.actor.state_dict(),
                    'critic_weights': self.critic.state_dict(),
                    'actor_optim': self.actor_optim.state_dict(),
                    'critic_optim': self.critic_optim.state_dict()}, path)

    def load(self, path):
        model = torch.load(path)
        self.actor.load_state_dict(model['actor_weights'])
        self.critic.load_state_dict(model['critic_weights'])
        self.actor_optim.load_state_dict(model['actor_optim'])
        self.critic_optim.load_state_dict(model['critic_optim'])

class MCT:
    """
    A Monte-Carlo tree
    """
    def __init__(self):
        self.action = None
        self.n_plays = 0
        self.n_wins = 0
        self.children = []
    
    def go_down(self, state):
        if len(children) > 0:
            child = random.choice(self.children)
            

class MCTSAgent(Agent):
    """
    An agent playing randomly
    """

    def __init__(self):
        super().__init__()
        self.mct = None

    def play(self, state, env, player_id):
        if self.mct is None:
            self.mct = MCT()
            self.mct.children = env.valid_actions()
        start_t = time.time()
        while time.time() - start_t < 2:
            
        new_state, _, done, _ = env.step(action)
        return new_state, done
