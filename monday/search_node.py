import math

def puct_value(node, parent_explore_count, uct_c):
    """Returns the PUCT value of child."""
    if node.outcome is not None:
      return (node.outcome[node.player] +
            uct_c * node.prior * math.sqrt(parent_explore_count) /
            (node.history.explore_count + 1))

    return (node.history.value() +
            uct_c * node.prior * math.sqrt(parent_explore_count) /
            (node.history.explore_count + 1))

class NodeHistory(object):
  def __init__(self):
    self.explore_count = 0
    self.total_reward = 0.0

  def visit(self, reward):
    self.explore_count += 1
    self.total_reward += reward

  def value(self):
    return self.explore_count and self.total_reward / self.explore_count

class SearchNode(object):
  def __init__(self, action, player, prior):
    self.action = action
    self.player = player
    self.prior = prior
    self.history = NodeHistory()
    self.outcome = None
    self.children = []

  def value(self, parent_explore_count, uct_c):
      return self.puct_value(self, parent_explore_count, uct_c)

  def uct_value(self, parent_explore_count, uct_c):
    """Returns the UCT value of child."""
    if self.outcome is not None:
      return self.outcome[self.player]

    if self.explore_count == 0:
      return float("inf")

    return self.history.value() + uct_c * math.sqrt(
        math.log(parent_explore_count) / self.history.explore_count)

  def puct_value(self, parent_explore_count, uct_c):
    """Returns the PUCT value of child."""
    if self.outcome is not None:
      return (self.outcome[self.player] +
            uct_c * self.prior * math.sqrt(parent_explore_count) /
            (self.history.explore_count + 1))

    return (self.history.value() +
            uct_c * self.prior * math.sqrt(parent_explore_count) /
            (self.history.explore_count + 1))

  def sort_key(self):
    """Returns the best action from this node, either proven or most visited.

    This ordering leads to choosing:
    - Highest proven score > 0 over anything else, including a promising but
      unproven action.
    - A proven draw only if it has higher exploration than others that are
      uncertain, or the others are losses.
    - Uncertain action with most exploration over loss of any difficulty
    - Hardest loss if everything is a loss
    - Highest expected reward if explore counts are equal (unlikely).
    - Longest win, if multiple are proven (unlikely due to early stopping).
    """
    return (0 if self.outcome is None else self.outcome[self.player],
            self.history.explore_count, self.history.total_reward)

  def best_child(self):
    """Returns the best child in order of the sort key."""
    return max(self.children, key=SearchNode.sort_key)

  def children_str(self, state=None):
    """Returns the string representation of this node's children.

    They are ordered based on the sort key, so order of being chosen to play.

    Args:
      state: A `pyspiel.State` object, to be used to convert the action id into
        a human readable format. If None, the action integer id is used.
    """
    return "\n".join([
        c.to_str(state)
        for c in reversed(sorted(self.children, key=SearchNode.sort_key))
    ])

  def to_str(self, state=None):
    """Returns the string representation of this node.

    Args:
      state: A `pyspiel.State` object, to be used to convert the action id into
        a human readable format. If None, the action integer id is used.
    """
    action = (
        state.action_to_string(state.current_player(), self.action)
        if state and self.action is not None else str(self.action))
    return ("{:>6}: player: {}, prior: {:5.3f}, value: {:6.3f}, sims: {:5d}, "
            "outcome: {}, {:3d} children").format(
                action, self.player, self.prior, self.history.value(), self.history.explore_count,
                ("{:4.1f}".format(self.outcome[self.player])
                 if self.outcome else "none"), len(self.children))

  def __str__(self):
    return self.to_str(None)

