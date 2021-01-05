from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time

import numpy as np
from search_node import NodeHistory, SearchNode

random_state = np.random.default_rng()
dirichlet_alpha = 0.1
dirichlet_epsilon = 0.25
uct_c = 2

def prior_with_noise(action_priors):
  noise = random_state.dirichlet([dirichlet_alpha] * len(action_priors))
  return [(a, (1 - dirichlet_epsilon) * p + dirichlet_epsilon * n) for (a, p), n in zip(action_priors, noise)]

# def policy_with_noise(policy):
#   noise = random_state.dirichlet([dirichlet_alpha] * len(policy))
#   return [(1 - dirichlet_epsilon) * p + dirichlet_epsilon * n for p, n in zip(policy, noise)]

def expand_node(az_evaluator, state, node, history_cache, is_root):
    player = state.current_player()
    obs = state.observation_tensor()
    mask = state.legal_actions_mask()
    actions = state.legal_actions()
    _, policy = az_evaluator._inference(obs, mask)
    priors = [(action, policy[action]) for action in actions]
    if is_root:
        priors = prior_with_noise(priors)

    state_key = tuple(obs)
    try:
        cached_histories = history_cache[state_key]
    except:
        cached_histories = [NodeHistory() for _ in priors]
        history_cache[state_key] = cached_histories

    node.children = [SearchNode(action, player, prior, history) for (action, prior), history in zip(priors, cached_histories)]
    random_state.shuffle(node.children)

def _find_leaf(az_evaluator, state, root, history_cache, is_first_expansion):
    visit_path = [root]
    working_state = state.clone()
    current_node = root
    is_prev_a_simultaneous = False
    is_root = is_first_expansion

    while not working_state.is_terminal() and (current_node.history.explore_count > 0 or is_prev_a_simultaneous):
        if len(current_node.children) == 0 or (is_root):
            expand_node(az_evaluator, working_state, current_node, history_cache, is_root)

        # print("children", current_node.children)
        chosen_child = max(current_node.children, key=lambda c: SearchNode.puct_value(c, current_node.history.explore_count, uct_c))

        is_prev_a_simultaneous = working_state.is_simultaneous_node()
        is_root = is_root and is_prev_a_simultaneous

        working_state.apply_action(chosen_child.action)
        current_node = chosen_child
        visit_path.append(current_node)

    return visit_path, working_state

def mcts_search(az_evaluator, state, root, history_cache, is_training=True):
    for n in range(501):
        visit_path, working_state = _find_leaf(az_evaluator, state, root, history_cache, n == 0 and is_training)

        if working_state.is_terminal():
            returns = working_state.returns()
            # visit_path[-1].history.outcome = returns
            solved = True
        else:
            obs = working_state.observation_tensor()
            mask = working_state.legal_actions_mask()
            returns, _ = az_evaluator._inference(obs, mask)
            solved = False

        for node in reversed(visit_path):
            node.history.visit(returns[node.player])

            if solved and node.children:
                player = node.children[0].player
                # If any have max utility (won?), or all children are solved,
                # choose the one best for the player choosing.
                best = None
                all_solved = True
                for child in node.children:
                    if child.history.outcome is None:
                        all_solved = False
                    elif best is None or child.history.outcome[player] > best.history.outcome[player]:
                        best = child
                if (best is not None and (all_solved or best.history.outcome[player] == 1)):
                    node.history.outcome = best.history.outcome
                else:
                    solved = False
