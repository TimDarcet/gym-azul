#!/usr/bin/env python3

import gym
from gym import error, spaces, utils
import logging.config
import pkg_resources
import cfg_load
from gym_azul.classes.player import Player
from gym_azul.classes.tile import Tile

# Constants
N_REPOS = 7
N_PLAYERS = 3


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
        self.repos = [{t: 0 for t in Tile} for _ in range(N_REPOS + 1)]  # 0 is the center repo
        self.players = [Player() for _ in range(N_PLAYERS)]
        # Action space
        self.action_space = spaces.Dict({
            "take": spaces.Discrete(N_REPOS + 1),
            "put": spaces.Discrete(5 + 1)
        })
        # Observation space
        self.observation_space = spaces.Dict({
            "you": player_space,
            "others": spaces.Tuple(tuple([player_space] * (N_PLAYERS - 1))),
            "repos": spaces.Tuple(tuple(repo_space for _ in range(N_REPOS + 1)))
        })

    def step(self, action):
        print(action)

    def reset(self):
        turn_to_play = 0
        self.repos = [{t: 0 for t in Tile} for _ in range(N_REPOS + 1)]  # 0 is the center repo
        self.players = [Player() for _ in range(N_PLAYERS)]

    def render(self, mode='human'):
        pass

    def close(self):
        pass
