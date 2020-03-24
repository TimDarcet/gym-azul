import gym
import torch.optim as optim
import numpy as np
from agent import Agent
from gym_azul.classes.box_wrapper import BoxWrapper

N_EPISODES = 10
ACTOR_LR = 1e-3
CRITIC_LR = 1e-4
HIDDEN_DIM = 32
GAMMA = 0.9
N_AGENTS = 2
N_REPOS = 7
UPDATE_EVERY = 5

env = BoxWrapper(gym.make("gym_azul:azul-v0", n_players=N_AGENTS, n_repos=N_REPOS))

state_dim = env.observation_space.shape[0]
action_dim = 5 * 6 * (N_REPOS + 1)

agents = []
for i in range(N_AGENTS):
    actor_optim = optim.Adam
    critic_optim = optim.Adam
    agents.append(Agent(state_dim, action_dim, HIDDEN_DIM, actor_optim, critic_optim, ACTOR_LR, CRITIC_LR, GAMMA))

for ep in range(N_EPISODES):
    state = env.reset()
    done = False
    counter = 0
    print('Game {}/{}'.format(ep, N_EPISODES))
    while not done:
        update = counter % UPDATE_EVERY
        counter += 1
        print('\rMove {}'.format(counter), end='')
        for id, agent in enumerate(agents):
            state, done = agent.play(state, env, id)
            if update:
                agent.update()
        print('')

