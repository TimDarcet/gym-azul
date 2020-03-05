import gym
import torch.optim as optim
from agent import Agent

N_EPISODES = 10
ACTOR_LR = 1e-3
CRITIC_LR = 1e-4
HIDDEN_DIM = 32
GAMMA = 0.9
N_AGENTS = 3
UPDATE_EVERY = 5

env = gym.make("gym_azul:azul-v0")

state_dim = env.observation_space.shape
action_dim = env.action_space.n

agents = []
for i in range(N_AGENTS):
    actor_optim = optim.Adam(lr=ACTOR_LR)
    critic_optim = optim.Adam(lr=CRITIC_LR)
    agents.append(Agent(state_dim, action_dim, HIDDEN_DIM, actor_optim, critic_optim, GAMMA))

for ep in range(N_EPISODES):
    state = env.reset()
    done = False
    counter = 0
    while not done:
        update = counter % UPDATE_EVERY
        counter += 1
        for id, agent in enumerate(agents):
            state, done = agent.play(state, env, id)
            if update:
                agent.update()

