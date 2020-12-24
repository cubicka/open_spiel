# is_terminal
# current_player
# observation_tensor
# legal_action_mask
# action to string
# is_simultaneous_node
# apply_action
# returns
# clone
# legal_actions
import numpy as np
import copy

def empty_copy(obj):
    class Empty(obj.__class__):
        def __init__(self): pass
    newcopy = Empty()
    newcopy.__class__ = obj.__class__
    return newcopy
    
_NUM_ROWS = 3
_NUM_COLS = 3
_NUM_CELLS = 9
_NUM_PLAYERS = 2

def _line_value(line):
  """Checks a possible line, returning the winning symbol if any."""
  if all(line == "x") or all(line == "o"):
    return line[0]


class TicTacToeState(object):
  """A python-only version of the Tic-Tac-Toe state.

  This class implements all the pyspiel.State API functions. Please see spiel.h
  for more thorough documentation of each function.

  Note that this class does not inherit from pyspiel.State since pickle
  serialization is not possible due to what is required on the C++ side
  (backpointers to the C++ game object, which we can't get from here).
  """

  def __init__(self, game):
    self._cur_player = 0
    self._winner = None
    self._is_terminal = False
    self._board = np.full((_NUM_ROWS, _NUM_COLS), ".")

  # Helper functions (not part of the OpenSpiel API).

  def _coord(self, move):
    return (move // _NUM_COLS, move % _NUM_COLS)

  def _line_exists(self):
    """Checks if a line exists, returns "x" or "o" if so, and None otherwise."""
    return (_line_value(self._board[0]) or _line_value(self._board[1]) or
            _line_value(self._board[2]) or _line_value(self._board[:, 0]) or
            _line_value(self._board[:, 1]) or _line_value(self._board[:, 2]) or
            _line_value(self._board.diagonal()) or
            _line_value(np.fliplr(self._board).diagonal()))

  # OpenSpiel (PySpiel) API functions are below. These need to be provided by
  # every game. Some not-often-used methods have been omitted.

  def current_player(self):
    return -1 if self._is_terminal else self._cur_player

  def legal_actions(self):
    """Returns a list of legal actions, sorted in ascending order.

    Args:
      player: the player whose legal moves

    Returns:
      A list of legal moves, where each move is in [0, num_distinct_actions - 1]
      at non-terminal states, and empty list at terminal states.
    """

    if self.is_terminal():
      return []

    actions = []
    for action in range(_NUM_CELLS):
        if self._board[self._coord(action)] == ".":
            actions.append(action)
    return actions

  def observation_tensor(self):
    obs = [
        1 if self._cur_player == 0 else -1,
        1 if self._cur_player == 1 else -1,
    ]

    obs.extend([1 if self._board[self._coord(x)]=='x' else 0 for x in range(_NUM_CELLS)])
    obs.extend([1 if self._board[self._coord(x)]=='o' else 0 for x in range(_NUM_CELLS)])
    return np.array(obs)

  def legal_actions_mask(self, player=None):
    """Get a list of legal actions.

    Args:
      player: the player whose moves we want; defaults to the current player.

    Returns:
      A list of legal moves, where each move is in [0, num_distinct_actios - 1].
      Returns an empty list at terminal states, or if it is not the specified
      player's turn.
    """
    if self.is_terminal():
      return []
    
    action_mask = [0] * _NUM_CELLS
    for action in self.legal_actions():
        action_mask[action] = 1
    return action_mask

  def apply_action(self, action):
    """Applies the specified action to the state."""
    self._board[self._coord(action)] = "x" if self._cur_player == 0 else "o"
    is_done = True
    for action in range(_NUM_CELLS):
        if self._board[self._coord(action)] == ".":
            is_done = False
    if self._line_exists():
      self._is_terminal = True
      self._winner = self._cur_player
    elif is_done:
      self._is_terminal = True
    else:
      self._cur_player = 1 - self._cur_player

  def action_to_string(self, arg0, arg1=None):
    """Action -> string. Args either (player, action) or (action)."""
    player = self.current_player() if arg1 is None else arg0
    action = arg0 if arg1 is None else arg1
    row, col = self._coord(action)
    return "{}({},{})".format("x" if player == 0 else "o", row, col)

  def is_terminal(self):
    return self._is_terminal

  def returns(self):
    if self.is_terminal():
      if self._winner == 0:
        return [1.0, -1.0]
      elif self._winner == 1:
        return [-1.0, 1.0]
    return [0.0, 0.0]

  def is_simultaneous_node(self):
    return False

  def __str__(self):
    return "\n".join("".join(row) for row in self._board)

  def clone(self):
    # return copy.deepcopy(self)
    a_copy = empty_copy(self)
    a_copy._cur_player = self._cur_player
    a_copy._winner = self._winner
    a_copy._is_terminal = self._is_terminal
    a_copy._board = [row[:] for row in self._board]
    # a_copy._board = copy.deepcopy(self._board)
    return a_copy

# new_initial_state
# num_distinct_actions
# num_players
# observation_tensor_shape

class TicTacToeGame(object):
  """A python-only version of the Tic-Tac-Toe game.

  This class implements all the pyspiel.Gae API functions. Please see spiel.h
  for more thorough documentation of each function.

  Note that this class does not inherit from pyspiel.Game since pickle
  serialization is not possible due to what is required on the C++ side
  (backpointers to the C++ game object, which we can't get from here).
  """

  def __init__(self):
    pass

  def new_initial_state(self):
    return TicTacToeState(self)

  def num_distinct_actions(self):
    return _NUM_CELLS

  def num_players(self):
    return _NUM_PLAYERS

  def observation_tensor_shape(self):
      return np.array([20])


