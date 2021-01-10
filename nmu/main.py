import numpy as np
from model import MuModel
from mcts import mcts_search, print_tree
from tictactoe import TicTacToe
# from timeit import default_timer as timer
from selfplay import play

def print_observation(obs):
  print(np.where(obs != 0))

game = TicTacToe()
mu = MuModel(game.num_states(), game.num_actions())

observations, policies, actions, returns = play(mu, game.clone())
# batch = zip(observations, policies, actions)
state = game.clone()
for idx in range(len(observations)):
  print(state)
  # print("observations")
  # print(observations[idx])
  print_observation(observations[idx])
  # print("policies")
  print(policies[idx])
  # print("actions")
  print(actions[idx])
  state.apply_action(actions[idx])
  print("\n\n")

print(state)
print(returns)

# start = timer()
# while not game.is_terminal():
#   print("\n\n")
#   actions = game.legal_actions()
#   policy, root = mcts_search(mu, game.observation_tensor(), actions, 500)
#   print(policy)
#   print_tree(root, depth=2)
#   # act = np.random.choice(actions, p=policy[actions])
#   act = actions[0]
#   print(act)
#   game.apply_action(act)
#   print(game)
# end = timer()
# print(end - start)
# s0 = mu.ht2(game.observation_tensor())
# print(s0)
# policy, value = mu.ft2(s0)
# print(value)
# print(policy)

# s1, r0 = mu.gt2(s0, 4)
# print(r0)
# print(s1)
