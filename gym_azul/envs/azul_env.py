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
from random import shuffle


# Constants
# MEF: Hardcoded, and hardcoded in the agent's code too
N_REPOS = 7
N_PLAYERS = 3
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

    def __init__(self):
        self.__version__ = "0.0.1"
        logging.info("AzulEnv - Version {}".format(self.__version__))

        # Internal state
        self.turn_to_play = 0 # Whose turn is it to play
        self.repos = [OrderedDict(sorted({t: 0 for t in Tile}.items())) for _ in range(N_REPOS + 1)]  # 0 is the center repo
        self.players = [Player() for _ in range(N_PLAYERS)]
        # Action space
        self.action_space = spaces.Dict({
            "player_id": spaces.Discrete(N_PLAYERS),
            "take": spaces.Dict({
                "repo": spaces.Discrete(N_REPOS + 1),
                "color": spaces.Discrete(len(Tile))
            }),
            "put": spaces.Discrete(5 + 1)
        })
        # Observation space
        self.observation_space = spaces.Dict({
            "you": player_space,
            "others": spaces.Tuple(tuple([player_space] * (N_PLAYERS - 1))),
            "repos": spaces.Tuple(tuple(repo_space for _ in range(N_REPOS + 1)))
        })
        # self.fill_repos()

    def step(self, action):
        assert not self.ending_condition(), "Le jeu est terminé, décroche"
        assert 0 <= action["player_id"] < N_PLAYERS
        # The player ID is redundant, it is only used to check if the agent is properly working
        if action["player_id"] != self.turn_to_play:
            print("Wrong player ID")
            return self.invalid_action()
        p = self.players[self.turn_to_play]
        # Take the tiles
        repo = action["take"]["repo"]
        color = list(Tile)[action["take"]["color"]]
        if self.repos[repo][color] == 0:  # He took zero tile
            return self.invalid_action()
        n_tiles = self.repos[repo][color]
        # Remove all tiles from repo and put the tiles not taken in the center
        for t in Tile:
            if t != color and repo != 0:
                self.repos[0][t] += self.repos[repo][t]
            self.repos[repo][t] = 0
        # Place the tiles
        q_id = action["put"]
        valid_action, points_won = p.place_tile(color, n_tiles, q_id)
        if not valid_action:
            return self.invalid_action()
        # End turn
        self.turn_to_play = (self.turn_to_play + 1) % N_PLAYERS
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
        assert self.all_repos_empty()
        turn_to_play = 0
        self.repos = [OrderedDict(sorted({t: 0 for t in Tile}.items())) for _ in range(N_REPOS + 1)]  # 0 is the center repo
        self.players = [Player() for _ in range(N_PLAYERS)]
        self.fill_repos()
        return self.observe()

    def render(self, mode='human'):
        print("Turn to play:", self.turn_to_play)
        for i in range(N_REPOS):
            print("Repo {}: ".format(i), end='')
            for t, v in self.repos[i].items():
                for _ in range(v):
                    print(t.name, end=' ')
            print('')
        print()
        for i in range(N_PLAYERS):
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
        while len(all_tiles) >= 4 and next_repo < N_REPOS:
            shuffle(all_tiles)
            here_tiles = all_tiles[:4]
            all_tiles = all_tiles[4:]
            self.repos[next_repo] = OrderedDict(sorted({t: here_tiles.count(t) for t in Tile}.items()))
            next_repo += 1
        if next_repo < N_REPOS:
            self.repos[next_repo] = OrderedDict(sorted({t: all_tiles.count(t) for t in Tile}.items()))
            next_repo +=1
            while next_repo < N_REPOS:
                self.repos[next_repo] = OrderedDict(sorted({t: 0 for t in Tile}.items()))
                next_repo +=1
        self.repos[0] = OrderedDict(sorted({t: 0 for t in Tile}.items()))

    def all_repos_empty(self):
        return all(all(x == 0 for x in r.values()) for r in self.repos)

    def end_round(self):
        for p in self.players:
            p.end_round()
        self.fill_repos()

    def observe(self):
        """Return the state viewed from player self.turn_to_play + 1"""
        player_id = (self.turn_to_play + 1) % N_PLAYERS
        others = self.players[:player_id] + self.players[player_id + 1:]
        d = OrderedDict(sorted({
            "you": self.players[player_id].observe(),
            "others": tuple(p.observe() for p in others),
            "repos": tuple(self.repos)
        }.items()))
        return d

    def ending_condition(self):
        return any(any(all(l) for l in p.square) for p in self.players)

    def is_valid(self, action):
        action["player_id"] == self.turn_to_play
        not self.ending_condition()
        #TODO

    def invalid_action(self):
        return self.observe(), -100, self.ending_condition(), {}
