import numpy as np

random_state = np.random.RandomState()

def mcts_evaluation(state, player):
  """Returns evaluation on given state."""
  working_state = state.clone()
  while not working_state.is_terminal():
    # print('==?', working_state.legal_actions())
    action = random_state.choice(working_state.legal_actions())
    working_state.apply_action(action)

  return working_state.returns()

def mcts_prior(state):
  legal_actions = state.legal_actions()
  return [(action, 1.0 / len(legal_actions)) for action in legal_actions]
