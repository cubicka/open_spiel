import numpy as np
from utils.empty_copy import empty_copy

def is_player_win(board, player):
    pad = player*9
    return (all(board[pad:pad+3] == 1) or all(board[pad+3:pad+6] == 1) or all(board[pad+6:pad+9] == 1) or
        all(board[pad:pad+9:3] == 1) or all(board[pad+1:pad+9:3] == 1) or all(board[pad+2:pad+9:3] == 1) or
        all(board[pad+0:pad+9:4] == 1) or all(board[pad+2:pad+7:2] == 1))

def apply_action(board, player, action):
    board[player*9 + action] = 1

def legal_actions(board, player):
    return [x for x in range(9) if board[x] == 0 and board[x+9] == 0]

def observation_tensor(board, player):
    obs = [1, 0, 0, 1] if player == 0 else [0, 1, 1, 0]
    obs.extend(board)
    return obs

class TicTacToe(object):
    def __init__(self):
        self._is_terminal = False
        self._cur_player = 0
        self._board = np.zeros([18], dtype=int)
        self._winner = None

    def num_players(self):
        return 2

    def num_actions(self):
        return 9

    def num_states(self):
        return 22

    def is_simultaneous_node(self):
        return False

    def is_terminal(self):
        return self._is_terminal

    def returns(self):
        if self.is_terminal():
            if self._winner == 0:
                return [1.0, -1.0]
            elif self._winner == 1:
                return [-1.0, 1.0]
        return [0.0, 0.0]

    def current_player(self):
        return -1 if self._is_terminal else self._cur_player

    def observation_tensor(self):
        return observation_tensor(self._board, self._cur_player)

    def legal_actions(self):
        if self.is_terminal():
            return []

        return legal_actions(self._board, self._cur_player)

    def legal_actions_mask(self):
        action_mask = np.zeros([9])
        for action in self.legal_actions():
            action_mask[action] = 1
        return action_mask

    def apply_action(self, action):
        apply_action(self._board, self._cur_player, action)
        if is_player_win(self._board, self._cur_player):
          self._is_terminal = True
          self._winner = self._cur_player
        elif len(self.legal_actions()) == 0:
          self._is_terminal = True
        else:
          self._cur_player = 1 - self._cur_player


    def action_to_string(self, player, action):
        return "{}({},{})".format("x" if player == 0 else "o", action // 3, action % 3)

    def __str__(self):
        return "\n".join("{}: {}".format(player, 
            ", ".join("({},{})".format(x//3, x%3) for x in range(9) if self._board[x+player*9] == 1))
            for player in (0,1))

    def clone(self):
        a_copy = empty_copy(self)
        a_copy._cur_player = self._cur_player
        a_copy._winner = self._winner
        a_copy._is_terminal = self._is_terminal
        a_copy._board = self._board.copy()
        return a_copy
