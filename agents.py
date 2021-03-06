import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import random
import copy
from gym_azul.classes.box_wrapper import convertor
import gym

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
            print('Move format: type "i j k" where i is a repo ID, j is a color ID and k is a queue ID.')
            print('Color IDs: BLUE = 0 YELLOW = 1 RED = 2 BLACK = 3 CYAN = 4')
            try:
                repo, color, queue = input('Your move: ').split()
                repo, color, queue = int(repo), int(color), int(queue)
                assert 0 <= repo < env.n_repos and 0 <= color < 5 and 0 <= queue < 6
                action = {'player_id': player_id, 'take': {'repo': repo, 'color': color}, 'put': queue}
                new_state, _, done, _ = env.step(action)
            except (ValueError, AssertionError):
                print('Your action is invalid, please try again')
            else:
                played = True
        return new_state, done


class A2CAgent(Agent):
    """
    An agent learning with A2C
    """
    def __init__(self, env, hidden_dim=256, actor_optim=None, critic_optim=None, actor_lr= 1e-2, critic_lr=1e-3, gamma=0.9, nb_channels=1):
        super().__init__()

        self.box_convertor = convertor(env.observation_space)

        # some info
        self.hidden_dim = hidden_dim
        self.action_dim = 5 * 6 * (env.n_repos + 1)
        self.state_dim = self.box_convertor.out_space.shape[0]
        self.gamma = gamma

        # initializes networks
        self.actor = Actor(self.state_dim, self.action_dim, hidden_dim)
        self.critic = Critic(self.state_dim, hidden_dim)

        # if nb_channels > 1, the agent plays for several players (or in several games) and the histories must be tracked parallely
        self.nb_channels = nb_channels

        # to able or disable learning
        self.learning = True

        # creates empty reward, value and log odds records
        self.rewards = [[] for _ in range(self.nb_channels)]
        self.values = [[] for _ in range(self.nb_channels)]
        self.logodds = [[] for _ in range(self.nb_channels)]

        # initializes optimizers
        if not actor_optim:
            actor_optim = optim.Adam
        if not critic_optim:
            critic_optim = optim.Adam
        self.actor_optim = actor_optim(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = critic_optim(self.critic.parameters(), lr=critic_lr)

    def play(self, state, env, player_id):
        state = preprocess(self.box_convertor(state))  # converts into tensor
        distr = self.actor(state)
        action = np.random.choice(list(range(self.action_dim)), p=distr.detach().numpy())
        env.set_tolerant(True)  # so that the env samples a random action if the action is invalid
        new_state, reward, done, _ = env.step(postprocess(action, player_id, env.n_repos))
        env.set_tolerant(False)
        if self.learning:
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
    exploration_rate = np.sqrt(2)
    id = 0
    def __init__(self):
        self.action = None
        self.n_plays = 0
        self.n_wins = 0
        self.children = []
        self.id = MCT.id
        MCT.id += 1

    def __str__(self):
        return 'Node {} [{}/{}]'.format(self.id, self.n_wins, self.n_plays)

    def go_down(self, env, player_id):
        if len(self.children) > 0:

            # already explored node, we choose a children and go down him

            # UCB formula for exploration - exploitation tradeoff
            same_side = player_id == env.turn_to_play
            if same_side:
                exploitation_weights = np.array([c.n_wins/(1 + c.n_plays) for c in self.children])
            else:
                # the more likely moves are the one that penalize player_id
                exploitation_weights = np.array([(c.n_plays - c.n_wins) / (1 + c.n_plays) for c in self.children])
            exploration_weights = np.array([MCT.exploration_rate * np.sqrt(np.log(self.n_plays)/(1 + c.n_plays)) for c in self.children])
            weights = exploitation_weights + exploration_weights

            child = random.choices(self.children, weights=weights)[0]

            print(self.id, env.valid_actions() == [c.action for c in self.children])

            _, _, done, _ = env.step(child.action)
            win = child.go_down(env, player_id)
            self.n_plays += 1
            if win:
                self.n_wins += 1
            return win
        else:
            # no child: either it was never explored, or it is terminal
            done = env.ending_condition()
            if done:
                return player_id == env.get_winner()

            # it was not explored
            for act in env.valid_actions():
                self.children.append(MCT())
                self.children[-1].action = act

            child = random.choice(self.children)
            env.step(child.action)

            done = env.ending_condition() # if the child is terminal, don't further simulate
            while not done:
                action = env.sample_action()
                _, _, done, _ = env.step(action)
            win = player_id == env.get_winner()[0]
            child.n_plays += 1
            self.n_plays += 1
            if win:
                child.n_wins += 1
                self.n_wins += 1
            return win


class MCTSAgent(Agent):
    """
    An agent using Monte-Carlo Tree Search. Works only for 2 players !
    """

    def __init__(self, timeout=0.3):
        super().__init__()
        self.mct = MCT()
        self.timeout = timeout
        self.dirty = False  # if the internal state does not match the one that is being given

    def play(self, state, env, player_id):
        # look for the state that is given in our inner tree
        if self.dirty:  #TODO: go down the tree while the state does not match to support games with >2 players
            for child in self.mct.children:
                dummy_env = copy.deepcopy(self.mct.env)
                dummy_env.step(child.action)
                if dummy_env.observe() == state: #TODO: is this an efficient comparison ?
                    self.mct = child
                    break
            else:
                raise ValueError("Couldn't find the action that led to this env")

        # repeatedly expand the tree
        start_t = time.time()
        while time.time() - start_t < self.timeout:
            dummy_env = copy.deepcopy(env)  # an env to performs actions and state computations
            self.mct.go_down(dummy_env, player_id)
        # once tree is built, select a move. Heuristic is best move = more visited move
        child = max(self.mct.children, key=lambda n: n.n_plays)
        new_state, _, done, _ = env.step(child.action)
        self.mct = child
        self.mct.env = copy.deepcopy(env) # to be able to look for the new state later
        self.dirty = True
        return new_state, done
