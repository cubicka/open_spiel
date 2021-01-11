import numpy as np
from mcts import mcts_search, print_tree

def play(mu, game):
    observations = []
    policies = []
    actions = []
    values = []
    players = []
    while not game.is_terminal():
        # print(game)
        obs = game.observation_tensor()
        cur_player = game.current_player()
        legal_actions = game.legal_actions()

        observations.append(obs)
        players.append(cur_player)
        hs = mu.ht(obs)

        policy, root = mcts_search(mu, obs, legal_actions, 500)
        policies.append(policy)
        values.append(root.value())

        next_action = np.random.choice(legal_actions, p=policy[legal_actions])
        game.apply_action(next_action)
        actions.append(next_action)

    return observations, players, actions, policies, values, game.returns()

def to_one_hot(x,n):
  ret = np.zeros([n])
  if x >= 0:
    ret[x] = 1.0
  return ret

def add_target(a_dim, targets, game_history):
    observations, actions, policies, returns = game_history
    n_obs = len(observations)
    # print("n_obs", n_obs)
    for idx in range(n_obs):
        obs_actions = actions[idx:idx+5]
        obs_policies = policies[idx:idx+5]
        # obs_returns = np.full(5, returns)

        if n_obs - idx < 5:
            n_policy = len(policies[0])
            n_missing = 5 - n_obs + idx

            obs_actions.extend([0] * n_missing)
            obs_policies.extend([[1.0/n_policy for _ in policies[0]]] * n_missing)

        targets.append((
            observations[idx], 
            to_one_hot(obs_actions[0], a_dim),
            to_one_hot(obs_actions[1], a_dim),
            to_one_hot(obs_actions[2], a_dim),
            to_one_hot(obs_actions[3], a_dim),
            obs_policies[0],
            obs_policies[1],
            obs_policies[2],
            obs_policies[3],
            obs_policies[4],
            [returns],
            [returns],
            [returns],
            [returns],
            [returns],
            # obs_returns[0],
            # obs_returns[1],
            # obs_returns[2],
            # obs_returns[3],
            # obs_returns[4],
        ))

def history_to_target(a_dim, game_history):
    observations, players, actions, policies, values, returns = game_history
    unique_players = np.unique(players)
    # print("players", players)

    targets = []
    for player in unique_players:
        player_idxs = [idx for idx, p in enumerate(players) if p == player]
        print("player_idxs", player_idxs)
        add_target(a_dim, targets, (
            # observations[players == player],
            # actions[players == player],
            # policies[players == player],
            # returns[player],
            [observations[idx] for idx in player_idxs],
            [actions[idx] for idx in player_idxs],
            [policies[idx] for idx in player_idxs],
            returns[player]
        ))

    return targets
