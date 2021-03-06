#!/usr/bin/env python3

import gym
from gym import error, spaces, utils
import logging.config
import pkg_resources
import cfg_load
from collections import OrderedDict
from gym_azul.classes.player import Player
from gym_azul.classes.tile import Tile
from gym_azul.classes.box_wrapper import BoxWrapper
from gym_azul.utils import *
from random import shuffle, choice

# Constants
MAX_PENALTY = 7

# Setup logging
path = 'config.yaml'  # always use slash in packages
filepath = pkg_resources.resource_filename('gym_azul', path)
config = cfg_load.load(filepath)
logging.config.dictConfig(config['LOGGING'])

player_space = spaces.Dict({
    "points": spaces.Discrete(300),
    "square": spaces.Tuple((
        spaces.MultiBinary(5),
        spaces.MultiBinary(5),
        spaces.MultiBinary(5),
        spaces.MultiBinary(5),
        spaces.MultiBinary(5)
    )),
    "queues": spaces.Tuple((
        spaces.Dict({
            "type": spaces.Discrete(6),
            "num": spaces.Discrete(2)
        }),
        spaces.Dict({
            "type": spaces.Discrete(6),
            "num": spaces.Discrete(3)
        }),
        spaces.Dict({
            "type": spaces.Discrete(6),
            "num": spaces.Discrete(4)
        }),
        spaces.Dict({
            "type": spaces.Discrete(6),
            "num": spaces.Discrete(5)
        }),
        spaces.Dict({
            "type": spaces.Discrete(6),
            "num": spaces.Discrete(6)
        })
    )),
    "penalties": spaces.Discrete(8)
})
repo_space = spaces.Dict({t: spaces.Discrete(21) for t in Tile})


class AzulEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, n_players=2):
        self.__version__ = "0.0.1"
        logging.info("AzulEnv - Version {}".format(self.__version__))

        self.n_players = n_players
        self.tolerant = False

        # initializes repositories
        if self.n_players == 2:
            self.n_repos = 5
        elif self.n_players == 3:
            self.n_repos = 7
        elif self.n_players == 4:
            self.n_repos = 9
        else:
            raise ValueError("n_players must be 2, 3 or 4")

        # Internal state
        self.turn_to_play = 0  # Whose turn is it to play
        self.repos = [OrderedDict(sorted({t: 0 for t in Tile}.items())) for _ in
                      range(self.n_repos + 1)]  # 0 is the center repo
        self.players = [Player() for _ in range(self.n_players)]
        # Action space
        self.action_space = spaces.Dict({
            "player_id": spaces.Discrete(self.n_players),
            "take": spaces.Dict({
                "repo": spaces.Discrete(self.n_repos + 1),
                "color": spaces.Discrete(len(Tile))
            }),
            "put": spaces.Discrete(5 + 1)
        })
        # Observation space
        self.observation_space = spaces.Dict({
            "you": player_space,
            "others": spaces.Tuple(tuple([player_space] * (self.n_players - 1))),
            "repos": spaces.Tuple(tuple(repo_space for _ in range(self.n_repos + 1)))
        })
        # self.fill_repos()

    def step(self, action):
        assert not self.ending_condition(), "Le jeu est terminé, décroche"
        assert 0 <= action["player_id"] < self.n_players, "Your player does not exists"
        # The player ID is redundant, it is only used to check if the agent is properly working

        # Some validity checks: player, repos and 'put' action
        if action["player_id"] != self.turn_to_play:  # player validity
            print("Wrong player ID: expected {} but got {}".format(self.turn_to_play, action["player_id"]))
            return self.invalid_action()

        p = self.players[self.turn_to_play]
        repo = action["take"]["repo"]
        color = list(Tile)[action["take"]["color"]]
        if self.repos[repo][color] == 0:  # repos validity: the player took zero tile
            print('Player tried to take non existent tiles')
            return self.invalid_action()
        n_tiles = self.repos[repo][color]
        q_id = action["put"]

        if not p.is_valid(color, q_id):  # 'put' action validity
            print('Player tried to put tiles somewhere forbidden')
            return self.invalid_action()

        # Update repos
        if repo != 0:
            # If the repo is not the center, remove all tiles from repo and put the tiles not taken in the center
            for t in Tile:
                if t != color:
                    self.repos[0][t] += self.repos[repo][t]
                self.repos[repo][t] = 0
        else:
            # If the repo is the center, just remove the chosen color
            self.repos[repo][color] = 0

        # Place the tiles
        valid_action, points_won = p.place_tile(color, n_tiles, q_id)
        # End turn
        self.turn_to_play = (self.turn_to_play + 1) % self.n_players
        state = self.observe()
        player_id = self.turn_to_play
        done = False
        if self.all_repos_empty():
            # Round ended
            self.end_round()
            if self.ending_condition():
                done = True
                self.close()
        return state, points_won, done, {}

    def reset(self):
        self.turn_to_play = 0
        self.repos = [OrderedDict(sorted({t: 0 for t in Tile}.items())) for _ in
                      range(self.n_repos + 1)]  # 0 is the center repo
        self.players = [Player() for _ in range(self.n_players)]
        self.fill_repos()
        return self.observe()

    def render(self, mode='human'):
        print("Turn to play:", self.turn_to_play)
        for i in range(self.n_repos):
            print("Repo {}: ".format(i), end='')
            for t, v in self.repos[i].items():
                for _ in range(v):
                    print(t.name, end=' ')
            print('')
        print()
        for i in range(self.n_players):
            print("Player {}:".format(i), str(self.players[i]), sep='\n')

    def close(self):
        # TODO
        pass

    def remaining_tiles(self):
        # MEF: This doesn't take into account tiles in repos
        # TODO: This doesn't take into account penalty tiles
        n_each = {t: 20 for t in Tile}
        for p in self.players:
            for qt, ql in p.queues:
                if qt is not None:
                    n_each[qt] -= ql
            for i in range(5):
                for j in range(5):
                    if p.square[i][j]:
                        n_each[tile_at(i, j)] -= 1
        return n_each

    def fill_repos(self):
        r_tiles = self.remaining_tiles()
        all_tiles = sum(([t] * v for t, v in r_tiles.items()), [])
        next_repo = 0
        while len(all_tiles) >= 4 and next_repo < self.n_repos:
            shuffle(all_tiles)
            here_tiles = all_tiles[:4]
            all_tiles = all_tiles[4:]
            self.repos[next_repo] = OrderedDict(sorted({t: here_tiles.count(t) for t in Tile}.items()))
            next_repo += 1
        if next_repo < self.n_repos:
            self.repos[next_repo] = OrderedDict(sorted({t: all_tiles.count(t) for t in Tile}.items()))
            next_repo += 1
            while next_repo < self.n_repos:
                self.repos[next_repo] = OrderedDict(sorted({t: 0 for t in Tile}.items()))
                next_repo += 1
        self.repos[0] = OrderedDict(sorted({t: 0 for t in Tile}.items()))

    def all_repos_empty(self):
        return all(all(x == 0 for x in r.values()) for r in self.repos)

    def end_round(self):
        for p in self.players:
            p.end_round()
        self.fill_repos()

    def observe(self):
        """Return the state viewed from player self.turn_to_play"""
        player_id = self.turn_to_play
        others = self.players[:player_id] + self.players[player_id + 1:]
        d = OrderedDict(sorted({
                                   "you": self.players[player_id].observe(),
                                   "others": tuple(p.observe() for p in others),
                                   "repos": tuple(self.repos)
                               }.items()))
        return d

    def ending_condition(self):
        """
        game ends if any of the players have a full line in their square
        """
        return any(any(all(line) for line in p.square) for p in self.players)

    def is_valid(self, action):
        action["player_id"] == self.turn_to_play
        not self.ending_condition()
        # TODO

    def invalid_action(self):
        """
        returns a penalized observation if the agent tried an invalid action.
        If self.tolerant, samples a random valid action (so that the game does not get stuck), if not it raises an error
        """
        if self.tolerant:
            replacement = self.sample_action()
            state, _, done, _ = self.step(replacement)
            return state, -100, done, {}
        else:
            raise ValueError("Action is invalid")

    def sample_action(self):
        """
        uniformly samples an action from all possible valid actions
        """
        if self.ending_condition():
            return None
        valid_actions = self.valid_actions()
        action = choice(valid_actions)  # uniform sampling
        return action

    def get_winner(self):
        """
        returns id and score of the winner
        """
        return max([(i, p.score) for i, p in enumerate(self.players)], key=lambda x: x[1])

    def set_tolerant(self, bool):
        self.tolerant = bool

    def valid_actions(self):
        """
        Return the list of valid actions
        """
        if self.ending_condition():
            return []
        player_id = self.turn_to_play
        p = self.players[player_id]
        valid_actions = []
        for repo in range(self.n_repos):
            for color_id in range(5):
                color = list(Tile)[color_id]
                if self.repos[repo][color] > 0:  # check that there are indeed tiles of this color in the repo
                    for queue in range(6):
                        if p.is_valid(color, queue):  # check that the player can put the color into this queue
                            valid_actions.append((repo, color_id, queue))
        return list(map(lambda act: {'player_id': player_id,
                                                    'take': {'repo': act[0], 'color': act[1]},
                                                    'put': act[2]},
                        valid_actions))

