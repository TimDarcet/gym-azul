import gym
import torch.optim as optim
from agent import Agent, RandomAgent
from gym_azul.classes.box_wrapper import BoxWrapper

# learning parameters
N_EPISODES = 1000
TRAINING_MODE = 'self'  # 'self' = self play,
# 'adversarial' = several networks learn against each others,
#  'random' = against a random baseline
GAMMA = 0.9  # weight of future rewards
UPDATE_EVERY = 5  # how many moves to play before updating
SAVE_EVERY = 100  # save a model every 1000 episodes

# learning rates
ACTOR_LR = 1e-3
CRITIC_LR = 1e-4

# network parameters
HIDDEN_DIM = 32

# game parameters
N_PLAYERS = 2

# paths
save_to = 'checkpoints'  # folder to save the models into
load_from = None  # to use pretrained agents

# ================== SET UP ENVIRONMENT ====================================

env = BoxWrapper(gym.make("gym_azul:azul-v0", n_players=N_PLAYERS))

state_dim = env.observation_space.shape[0]
action_dim = 5 * 6 * (env.n_repos + 1)

# =================== SET UP AGENTS ========================================

agents = []

if TRAINING_MODE == 'adversarial':
    # n learning agents
    for i in range(N_PLAYERS):
        actor_optim = optim.Adam
        critic_optim = optim.Adam
        agent = Agent(state_dim, action_dim, HIDDEN_DIM, actor_optim, critic_optim, ACTOR_LR, CRITIC_LR, GAMMA)
        if load_from:
            agent.load(load_from)
        agents.append(agent)

if TRAINING_MODE == 'self':
    # n times the same learning agent
    actor_optim = optim.Adam
    critic_optim = optim.Adam
    agent = Agent(state_dim, action_dim, HIDDEN_DIM, actor_optim, critic_optim, ACTOR_LR, CRITIC_LR, GAMMA,
                  nb_channels=N_PLAYERS)
    if load_from:
        agent.load(load_from)
    for i in range(N_PLAYERS):
        agents.append(agent)

if TRAINING_MODE == 'random':
    # 1 learning agent and n-1 random agents
    actor_optim = optim.Adam
    critic_optim = optim.Adam
    agent = Agent(state_dim, action_dim, HIDDEN_DIM, actor_optim, critic_optim, ACTOR_LR, CRITIC_LR, GAMMA)
    if load_from:
        agent.load(load_from)
    agents.append(agent)
    for i in range(N_PLAYERS - 1):
        agents.append(RandomAgent())

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
            if update:
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
