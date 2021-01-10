import numpy as np
from mcts import mcts_search, print_tree

def play(mu, game):
    observations = []
    policies = []
    actions = []
    while not game.is_terminal():
        # print(game)
        obs = game.observation_tensor()
        legal_actions = game.legal_actions()

        observations.append(obs)
        hs = mu.ht(obs)

        policy, root = mcts_search(mu, obs, legal_actions, 500)
        policies.append(policy)

        next_action = np.random.choice(legal_actions, p=policy[legal_actions])
        game.apply_action(next_action)
        actions.append(next_action)

    return observations, policies, actions, game.returns()