import gym
import os
import torch.optim as optim
from agents import A2CAgents, RandomAgent, HumanAgent
from gym_azul.classes.box_wrapper import BoxWrapper

# learning parameters
N_EPISODES = 150  # a few hours on a local pc
PLAYING_MODE = 'manual'
# 'self' -> self play,
# 'adversarial' -> several networks learn against each others,
# 'random' -> against a random baseline
# 'manual' -> against a human
TRAINING = True
GAMMA = 0.98  # weight of future rewards
UPDATE_EVERY = 15  # how many moves to play before updating (15 = update about twice a game)
SAVE_EVERY = 1000  # save a model every 1000 episodes

# learning rates
ACTOR_LR = 1e-3
CRITIC_LR = 1e-4

# network parameters
HIDDEN_DIM = 256

# game parameters
N_PLAYERS = 2

# paths
save_to = 'checkpoints_against_random'  # folder to save the models into
load_from = 'checkpoints/144999.pt'  # to use pretrained agents


# ================== SET UP ENVIRONMENT ====================================

env = BoxWrapper(gym.make("gym_azul:azul-v0", n_players=N_PLAYERS))

state_dim = env.observation_space.shape[0]
action_dim = 5 * 6 * (env.n_repos + 1)

if not os.path.exists(save_to):
    os.mkdir(save_to)

# =================== SET UP AGENTS ========================================

agents = []

if PLAYING_MODE == 'adversarial':
    # n learning agents
    for i in range(N_PLAYERS):
        actor_optim = optim.Adam
        critic_optim = optim.Adam
        agent = A2CAgents(state_dim, action_dim, HIDDEN_DIM, actor_optim, critic_optim, ACTOR_LR, CRITIC_LR, GAMMA)
        if load_from:
            agent.load(load_from)
        agents.append(agent)

if PLAYING_MODE == 'self':
    # n times the same learning agent
    actor_optim = optim.Adam
    critic_optim = optim.Adam
    agent = A2CAgents(state_dim, action_dim, HIDDEN_DIM, actor_optim, critic_optim, ACTOR_LR, CRITIC_LR, GAMMA,
                      nb_channels=N_PLAYERS)
    if load_from:
        agent.load(load_from)
    for i in range(N_PLAYERS):
        agents.append(agent)

if PLAYING_MODE == 'random':
    # 1 learning agent and n-1 random agents
    actor_optim = optim.Adam
    critic_optim = optim.Adam
    agent = A2CAgents(state_dim, action_dim, HIDDEN_DIM, actor_optim, critic_optim, ACTOR_LR, CRITIC_LR, GAMMA)
    if load_from:
        agent.load(load_from)
    agents.append(agent)
    for i in range(N_PLAYERS - 1):
        agents.append(RandomAgent())

if PLAYING_MODE == 'manual':
    # 1 agent and 1 human
    assert N_PLAYERS == 2
    actor_optim = optim.Adam
    critic_optim = optim.Adam
    agent = A2CAgents(state_dim, action_dim, HIDDEN_DIM, actor_optim, critic_optim, ACTOR_LR, CRITIC_LR, GAMMA)
    if load_from:
        agent.load(load_from)
    agents.append(agent)
    agents.append(HumanAgent())

# ==================== ACTUAL TRAINING ==========================================

for ep in range(N_EPISODES):
    state = env.reset()
    done = False
    counter = 0
    print('Game {}/{}'.format(ep + 1, N_EPISODES))
    while not done:
        update = not ((counter + 1) % UPDATE_EVERY)
        counter += 1
        for id, agent in enumerate(agents):
            if done: break
            state, done = agent.play(state, env, id)
            if update and TRAINING:
                agent.update()
    winner, score = env.get_winner()
    for id, agent in enumerate(agents):
        # has no meaning if agent is self-playing
        agent.next_game(winner == id)
    print('Game completed in {} moves. Agent {} won with score {}'.format(counter + 1, winner, score))
    if not ((ep + 1) % SAVE_EVERY):
        for agent in agents:
            agent.save('{}/{}.pt'.format(save_to, ep))

for id, agent in enumerate(agents):
    # has no meaning if agent is self-playing
    print('Agent {} stats: {}'.format(id, agent.stats))
