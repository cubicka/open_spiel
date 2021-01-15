import numpy as np
from mcts import mcts_search
import utils.logger as file_logger

def softmax(x):
  e_x = np.exp(x - max(x))
  return e_x / e_x.sum()

def play(mu, game, n_mcts_sim=500, with_noise=True):
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

        policy, root, _ = mcts_search(mu, obs, legal_actions, n_mcts_sim, with_noise)
        policies.append(policy)
        values.append(root.value())

        next_action = select_action(legal_actions, policy[legal_actions], 1)
        game.apply_action(next_action)
        actions.append(next_action)

    return observations, players, actions, policies, values, game.returns()

def explore(path, prevmu, mu, game, step, n_mcts_sim=500):
  with file_logger.FileLogger(path + '/log', step, True) as logger:
    # print_observation(logger, game.observation_tensor())
    # hs = mu.ht(game.observation_tensor())

    # pols, vals = mu.ft(hs)
    # soft_pol = softmax(pols)

    # logger.print(vals)
    # logger.print(pols)
    # logger.print(soft_pol)

    # a0 = select_action(game.legal_actions(), soft_pol[game.legal_actions()], 1)
    # gs, _ = mu.gt(hs, a0)

    # logger.print(hs)
    # logger.print(gs)
    # logger.print("\n\n")

    observations = []
    policies = []
    actions = []
    values = []
    players = []

    while not game.is_terminal():
        # print(game)
        logger.print("Initial state:\n{}".format(game))

        obs = game.observation_tensor()
        observations.append(obs)
        print_observation(logger, obs)

        cur_player = game.current_player()
        players.append(cur_player)
        legal_actions = game.legal_actions()
        hs = mu.ht(obs)

        # logger.print(legal_actions)
        policy, root, _ = mcts_search(mu, obs, legal_actions, n_mcts_sim, False)
        policies.append(policy)
        values.append(root.value())
        prevpolicy, prevroot, _ = mcts_search(prevmu, obs, legal_actions, n_mcts_sim, False)

        logger.print("Root ({}):".format(root.value()))
        logger.print(policy)
        print_tree(logger, root, prevroot, game)
        # logger.print(node.to_str(state, True))
        # logger.print()
        # logger.print("Children:")
        # logger.print("\n" + node.children_str(state))

        next_action = select_action(legal_actions, policy[legal_actions], 0) # np.random.choice(legal_actions, p=policy[legal_actions])
        actions.append(next_action)
        logger.print(game.action_to_string(game.current_player(), next_action))
        game.apply_action(next_action)

        logger.print("\n\n")

    logger.print("Final state:\n{}".format(game))
    logger.print("Returns:\n{}".format(game.returns()))

    # targets = history_to_target(game.num_actions(), (observations, players, actions, policies, values, game.returns()))
    # for i in range(len(targets)):
    #   obs, acts0, acts1, acts2, acts3, pols0, pols1, pols2, pols3, pols4, rets0, rets1, rets2, rets3, rets4 = zip(*targets[i:i+1])
    #   print(mu.check_loss(obs, acts0, acts1, acts2, acts3, pols0, pols1, pols2, pols3, pols4, rets0, rets1, rets2, rets3, rets4))

def select_action(actions, p, temp):
    if temp == 0:
        return actions[np.argmax(p)]
    return np.random.choice(actions, p=p)

def print_observation(logger, obs):
  logger.print(np.where(obs != 0))


def print_tree(logger, x, prevx, game, hist=[], depth=1):
  if x.visit_count != 0:
    delta = x.visit_count - prevx.visit_count if prevx is not None else x.visit_count
    dv = x.value() - prevx.value() if prevx is not None else x.value()
    logger.print("%.3f %4d %+4d %-16s %8.4f %+5.4f" % (x.prior, x.visit_count, delta, str(hist), x.value(), dv))
  if depth > 0:
    for c in x.children:
      prevchild = next((pc for pc in prevx.children if pc.action == c.action))
      print_tree(logger, c, prevchild, game, hist+[game.action_to_string(game.current_player(), c.action)], depth-1)

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
        # print("player_idxs", player_idxs)
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
