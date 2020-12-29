from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time

import numpy as np
from search_node import puct_value, NodeHistory, SearchNode

random_state = np.random.RandomState()
dirichlet_alpha = 1
dirichlet_epsilon = 0.25

def prior_with_noise(action_priors):
  noise = random_state.dirichlet([dirichlet_alpha] * len(action_priors))
  return [(a, (1 - dirichlet_epsilon) * p + dirichlet_epsilon * n) for (a, p), n in zip(action_priors, noise)]

def policy_with_noise(policy):
  noise = random_state.dirichlet([dirichlet_alpha] * len(policy))
  return [(1 - dirichlet_epsilon) * p + dirichlet_epsilon * n for p, n in zip(policy, noise)]

def create_children_nodes(player, action_priors, cached_histories):
  # For a new node, initialize its state, then choose a child as normal.
  # if with_noise: action_priors = prior_with_noise(action_priors)

  # Reduce bias from move generation order.
  return [SearchNode(action, player, prior, history) for (action, prior), history in zip(action_priors, cached_histories)]
  # random_state.shuffle(children)
  # return children

def _find_leaf(prior_fns, uct_c, root, state, history_cache):
  visit_path = [root]
  working_state = state.clone()
  current_node = root
  is_prev_a_simultaneous = False

  # print("\n===>>> Find leaf")
  # print("current node", current_node)
  while not working_state.is_terminal() and (current_node.history.explore_count > 0 or is_prev_a_simultaneous):
    if not current_node.children:
      state_key = np.expand_dims(working_state.observation_tensor(working_state.current_player()), 0).tobytes()
      priors = prior_fns[working_state.current_player()](working_state)
      if state_key in history_cache:
        cached_histories = history_cache[state_key]
      else:
        cached_histories = [NodeHistory() for action, prior in priors]
        history_cache[state_key] = cached_histories

      current_node.children = create_children_nodes(working_state.current_player(), priors, cached_histories)

    # hopeful_children = [c for c in current_node.children if c.outcome is None]
    # if len(hopeful_children) == 0 and is_prev_a_simultaneous:
    #   hopeful_children = current_node.children

    # print("visitpath")
    # for c in visit_path:
    #   print(c)
    # print("hopeful_children", is_prev_a_simultaneous, working_state)
    # for c in hopeful_children:
    #   print(c)

    chosen_child = max(current_node.children, key=lambda c: puct_value(c, current_node.history.explore_count, uct_c))
    # if len(hopeful_children) > 0:
    #   chosen_child = max(hopeful_children, key=lambda c: puct_value(c, current_node.history.explore_count, uct_c))
    # else:
    #   break

    is_prev_a_simultaneous = working_state.is_simultaneous_node()

    # print("action chosen", chosen_child)
    working_state.apply_action(chosen_child.action)
    current_node = chosen_child
    visit_path.append(current_node)

    # print("current node", current_node)

  # print("<<<=== end leaf\n")
  return visit_path, working_state

def mcts_search(evaluators, prior_fns, uct_c, state, history_cache):
  # root_player = state.current_player()
  root = SearchNode(None, state.current_player(), 1)
  opt_nums = len(state.legal_actions())
  # print("MCTS Walk begin")
  for n in range(opt_nums * 50):
  # for n in range(1000):
    visit_path, working_state = _find_leaf(prior_fns, uct_c, root, state, history_cache)
    # print("Visiting", n)
    # print(working_state)
    # for c in visit_path:
    #   print(c)

    if working_state.is_terminal():
      returns = working_state.returns()
      visit_path[-1].outcome = returns
      solved = True
    else:
      # eval_value = evaluators[working_state.current_player()](working_state, working_state.current_player())
      # returns = [-1*eval_value] * state.num_players()
      # returns[working_state.current_player()] = eval_value
      # returns = [evaluator(working_state, state.current_player()) for evaluator in evaluators]
      # returns = [evaluators[player](working_state, player) if player == working_state.current_player() else None for player in range(state.num_players())]
      returns = evaluators[working_state.current_player()](working_state, working_state.current_player())
      solved = False

    # print("Update Value", solved)
    # print(returns)
    for node in reversed(visit_path):
      # node.total_reward += returns[0 if node.player ==
      #                              pyspiel.PlayerId.CHANCE else node.player]
      # if returns[node.player] is not None:
      node.history.visit(returns[node.player])

      if solved and node.children:
        player = node.children[0].player
        # If any have max utility (won?), or all children are solved,
        # choose the one best for the player choosing.
        best = None
        all_solved = True
        # print("Got child done")
        for child in node.children:
          if child.outcome is None:
            all_solved = False
          elif best is None or child.outcome[player] > best.outcome[player]:
            best = child
        # print("Loop results", all_solved, best.outcome)
        # print(best)
        if (best is not None and (all_solved or best.outcome[player] == 1)):
          node.outcome = best.outcome
        else:
          solved = False

    #   # print(node)

    # if root.outcome is not None:
    #   break
  #   print("-------")
  # print("======= Done Update")
  # print("\n\n\n")

  return root
