import numpy as np
from model import MuModel
from mcts import mcts_search, print_tree
from tictactoe import TicTacToe
# from timeit import default_timer as timer
from selfplay import play, history_to_target

def print_observation(obs):
  print(np.where(obs != 0))

game = TicTacToe()
mu = MuModel(game.num_states(), game.num_actions())

for x in range(10):
  game_history = play(mu, game.clone())
  observations, players, actions, policies, values, returns = game_history
  # # batch = zip(observations, policies, actions)
  state = game.clone()
  for idx in range(len(observations)):
    print(state)
    # print("observations")
    # print(observations[idx])
    print_observation(observations[idx])
    # print("policies")
    print(values[idx])
    print(policies[idx])
    # print("actions")
    print(actions[idx])
    state.apply_action(actions[idx])
    print("\n\n")

  samples = history_to_target(game.num_actions(), game_history)
  obs, acts0, acts1, acts2, acts3, pols0, pols1, pols2, pols3, pols4, rets0, rets1, rets2, rets3, rets4 = zip(*samples)
  # print(samples)
  # for x in zip(samples):
  #   print(x)

  # for sample in samples:
  #   print("\n\n======= Sample")
  #   obs, acts, pols, rets = sample
  # print(obs)
  # print(acts0)
  # print(pols0)
  # print(rets0)
  print(mu.ht(obs[0]))
  _, policy_loss, value_loss, l2_loss = mu.train(obs, acts0, acts1, acts2, acts3, pols0, pols1, pols2, pols3, pols4, rets0, rets1, rets2, rets3, rets4)
  print("===Lossess===", policy_loss, value_loss, l2_loss)
  print("\n\n")
# print(state)
# print(returns)

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



# s0 = mu.ht(game.observation_tensor())
# print(s0)
# policy, value = mu.ft(s0)
# print(value)
# print(policy)

# s1, r0 = mu.gt(s0, 4)
# print(r0)
# print(s1)

# actions = game.legal_actions()
# policy, root = mcts_search(mu, game.observation_tensor(), actions, 500)
# print("\n\n")
# print(policy)
# print_tree(root, depth=2)
