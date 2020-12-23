from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time

import numpy as np
from search_node import puct_value, SearchNode

random_state = np.random.RandomState()
dirichlet_alpha = 1
dirichlet_epsilon = 0.25

def prior_with_noise(action_priors):
  noise = random_state.dirichlet([dirichlet_alpha] * len(action_priors))
  return [(a, (1 - dirichlet_epsilon) * p + dirichlet_epsilon * n) for (a, p), n in zip(action_priors, noise)]

def create_children_nodes(player, action_priors, with_noise):
  # For a new node, initialize its state, then choose a child as normal.
  if with_noise: action_priors = prior_with_noise(action_priors)

  # Reduce bias from move generation order.
  random_state.shuffle(action_priors)
  return [SearchNode(action, player, prior) for action, prior in action_priors]

def _find_leaf(prior_fns, uct_c, root, state, with_noise=False):
  visit_path = [root]
  working_state = state.clone()
  current_node = root
  is_prev_a_simultaneous = False
  root_noise = with_noise

  while not working_state.is_terminal() and (current_node.explore_count > 0 or is_prev_a_simultaneous):
    if not current_node.children:
      priors = prior_fns[working_state.current_player()](working_state)
      children = create_children_nodes(working_state.current_player(), priors, root_noise)

      if is_prev_a_simultaneous:
        for c in visit_path[-2].children:
          c.children = children
      else:
        current_node.children = children

    hopeful_children = [c for c in current_node.children if c.outcome is None]
    if len(hopeful_children) == 0 and is_prev_a_simultaneous:
      hopeful_children = current_node.children

    # print("current node", current_node)
    # print("visitpath")
    # for c in visit_path:
    #   print(c)
    # print("hopeful_children", is_prev_a_simultaneous, working_state)
    # for c in hopeful_children:
    #   print(c)

    if len(hopeful_children) > 0:
      chosen_child = max(hopeful_children, key=lambda c: puct_value(c, current_node.explore_count, uct_c))
    else:
      break

    is_prev_a_simultaneous = working_state.is_simultaneous_node()
    if not is_prev_a_simultaneous:
      root_noise = False

    working_state.apply_action(chosen_child.action)
    current_node = chosen_child
    visit_path.append(current_node)

    # print("action chosen", chosen_child)

  return visit_path, working_state

def mcts_search(evaluators, prior_fns, uct_c, with_noise, state):
  root_player = state.current_player()
  root = SearchNode(None, state.current_player(), 1)
  opt_nums = len(state.legal_actions())
  # print("MCTS Walk begin")
  for n in range(opt_nums * 50):
    visit_path, working_state = _find_leaf(prior_fns, uct_c, root, state, with_noise)
    # print("Visiting", n)
    # print(working_state)
    # for c in visit_path:
    #   print(c)

    if working_state.is_terminal():
      returns = working_state.returns()
      visit_path[-1].outcome = returns
      solved = True
    else:
      returns = [evaluator(working_state) for evaluator in evaluators]
      solved = False

    # print("Update Value", solved)
    # print(returns)
    for node in reversed(visit_path):
      # node.total_reward += returns[0 if node.player ==
      #                              pyspiel.PlayerId.CHANCE else node.player]
      node.total_reward += returns[node.player]
      node.explore_count += 1

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

      # print(node)

    if root.outcome is not None:
      break
  #   print("-------")
  # print("======= Done Update")
  # print("\n\n\n")

  return root
