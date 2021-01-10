import numpy as np
from utils.empty_copy import empty_copy
from numba import jit
# from collections import namedtuple
# from typings import NamedTuple

# TicTacToeState = namedtuple("TicTacToeState", '_is_terminal _cur_player _winner _board')
# class TicTacToeState(NamedTuple)

@jit(nopython=True)
def is_player_win(is_terminal, cur_player, winner, board):
    pad = cur_player*9
    return (np.all(board[pad:pad+3] == 1) or np.all(board[pad+3:pad+6] == 1) or 
        np.all(board[pad+6:pad+9] == 1) or np.all(board[pad:pad+9:3] == 1) or 
        np.all(board[pad+1:pad+9:3] == 1) or np.all(board[pad+2:pad+9:3] == 1) or
        np.all(board[pad+0:pad+9:4] == 1) or np.all(board[pad+2:pad+7:2] == 1))

@jit(nopython=True)
def apply_action(action, is_terminal, cur_player, winner, board):
    board[cur_player*9 + action] = 1
    if is_player_win(is_terminal, cur_player, winner, board):
        winner = cur_player
        is_terminal = True
    elif board.sum() == 9:
        is_terminal = True
    else:
        cur_player = 1 - cur_player
    return is_terminal, cur_player, winner, board

@jit(nopython=True)
def legal_actions(is_terminal, cur_player, winner, board):
    return np.array([x for x in range(9) if board[x] == 0 and board[x+9] == 0])

# @jit(nopython=True)
def observation_tensor(is_terminal, cur_player, winner, board):
    obs = np.zeros(22, dtype=int)
    if cur_player == 0:
        obs[0] = obs[3] = 1
    else:
        obs[1] = obs[2] = 1
    obs[4:] = board[:]
    return obs

class TicTacToe(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self._is_terminal = False
        self._cur_player = 0
        self._winner = -1
        self._board = np.zeros(18, dtype=int)

    def clone(self):
        a_copy = empty_copy(self)
        a_copy._cur_player = self._cur_player
        a_copy._winner = self._winner
        a_copy._is_terminal = self._is_terminal
        a_copy._board = self._board.copy()
        return a_copy

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
        return observation_tensor(self._is_terminal, self._cur_player, self._winner, self._board)

    def legal_actions(self):
        if self.is_terminal():
            return []

        return legal_actions(self._is_terminal, self._cur_player, self._winner, self._board)

    def legal_actions_mask(self):
        action_mask = np.zeros([9])
        for action in self.legal_actions():
            action_mask[action] = 1
        return action_mask

    def apply_action(self, action):
        self._is_terminal, self._cur_player, self._winner, self._board = apply_action(action, self._is_terminal, self._cur_player, self._winner, self._board)

    def action_to_string(self, player, action):
        return "{}({},{})".format("x" if player == 0 else "o", action // 3, action % 3)

    def __str__(self):
        # return "\n".join("{}: {}".format(player, 
        #     ", ".join("({},{})".format(x//3, x%3) for x in range(9) if self._board[x+player*9] == 1))
        #     for player in (0,1))
        def getc(idx):
            if self._board[idx] == 1: return 'x'
            elif self._board[idx+9] == 1: return 'o'
            else: return '.'

        return "\n".join("".join(getc(row*3+col) for col in range(3)) for row in range(3))
        # for row in range(3):
        #     for col in range(3):
        #         idx = row*3 + col
        #         if self._board[idx] == 1: 'x'
        #         elif self._board[idx+9] == 1: print 'o'
        #         else: print '.'
            
