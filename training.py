import gym
import torch.optim as optim
from agent import Agent, RandomAgent
from gym_azul.classes.box_wrapper import BoxWrapper

# learning parameters
N_EPISODES = 10
UPDATE_EVERY = 5  # how many moves to play before updating
TRAINING_MODE = 'self'  # 'self' = self play, 'adversarial' = several networks learn against each others, 'random' = against a random baseline
GAMMA = 0.9  # weight of future rewards

# learning rates
ACTOR_LR = 1e-3
CRITIC_LR = 1e-4

# network parameters
HIDDEN_DIM = 32

# game parameters
N_PLAYERS = 2

env = BoxWrapper(gym.make("gym_azul:azul-v0", n_players=N_PLAYERS))

state_dim = env.observation_space.shape[0]
action_dim = 5 * 6 * (env.n_repos + 1)

agents = []

if TRAINING_MODE == 'adversarial':
    # n learning agents
    for i in range(N_PLAYERS):
        actor_optim = optim.Adam
        critic_optim = optim.Adam
        agents.append(Agent(state_dim, action_dim, HIDDEN_DIM, actor_optim, critic_optim, ACTOR_LR, CRITIC_LR, GAMMA))

if TRAINING_MODE == 'self':
    # n times the same learning agent
    actor_optim = optim.Adam
    critic_optim = optim.Adam
    agent = Agent(state_dim, action_dim, HIDDEN_DIM, actor_optim, critic_optim, ACTOR_LR, CRITIC_LR, GAMMA, nb_channels=N_PLAYERS)
    for i in range(N_PLAYERS):
        agents.append(agent)

if TRAINING_MODE == 'random':
    # 1 learning agent and n-1 random agents
    actor_optim = optim.Adam
    critic_optim = optim.Adam
    agent = Agent(state_dim, action_dim, HIDDEN_DIM, actor_optim, critic_optim, ACTOR_LR, CRITIC_LR, GAMMA)
    agents.append(agent)
    for i in range(N_PLAYERS - 1):
        agents.append(RandomAgent())

for ep in range(N_EPISODES):
    state = env.reset()
    done = False
    counter = 0
    print('Game {}/{}'.format(ep, N_EPISODES))
    while not done:
        update = not (counter % UPDATE_EVERY + 1)
        counter += 1
        for id, agent in enumerate(agents):
            if done: break
            state, done = agent.play(state, env, id)
            if update:
                agent.update()
    print('\nGame completed in {} moves. '.format(counter + 1))


