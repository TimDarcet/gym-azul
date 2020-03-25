import gym
from agents import A2CAgent, RandomAgent, HumanAgent, MCTSAgent

N_PLAYERS = 2

env = gym.make("gym_azul:azul-v0", n_players=N_PLAYERS)

# define some agents
human = HumanAgent()
mcts = MCTSAgent()  # beware, only supports 2 players
random = RandomAgent()
a2c = A2CAgent(env, hidden_dim=256)
a2c_path = 'checkpoints/12999.pt'
a2c.learning = False
if a2c_path:
    a2c.load(a2c_path)

# which agents do you want to see playing
agents = [mcts, random]

# playing loop
state = env.reset()
done = False
while not done:
    for id, agent in enumerate(agents):
        if done: break
        state, done = agent.play(state, env, id)

winner, score = env.get_winner()
print('Agent {} won with score {}!'.format(winner, score))
