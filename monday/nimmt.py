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

def card_value(card):
    if (card == 55): return 7
    if (card%11 == 0): return 5
    if (card%10 == 0): return 3
    if (card%5 == 0): return 2
    return 1

def bull_value(cards):
    return sum([card_value(c) for c in cards])

def bull_to_return(val, minv, maxv):
    if val == minv: return 1
    if val == maxv: return -1
    return 0

class NimmtState(object):
    def __init__(self, game):
        self._cur_player = 0
        self._mode = 0
        self._num_players = np.random.randint(2, 11)
        
        shuffled_cards = [x for x in range(1, 105)]
        np.random.shuffle(shuffled_cards)
        self._hands = [set(shuffled_cards[player*10:(player+1)*10]) for player in range(self._num_players)]
        self._boards = [shuffled_cards[self._num_players*10+idx:self._num_players*10+idx+1] for idx in range(4)]
        self._bulls = [[] for _ in range(self._num_players)]
        self._stacks = []
        self._stack_idx = 0

        self._is_terminal = False

    def num_players(self):
        return self._num_players

    def current_player(self):
        return -1 if self._is_terminal else self._cur_player

    def is_simultaneous_node(self):
        return self._mode == 0 and self._cur_player < self._num_players - 1

    def is_terminal(self):
        return self._is_terminal

    def returns(self):
        if self.is_terminal():
            values = [bull_value(cards) for cards in self._bulls]
            best_value = min(values)
            worst_value = max(values)
            return list(map(lambda x: bull_to_return(values[x], best_value, worst_value), range(self._num_players)))
            # return [1.0 if values[x] == best_value else -1.0 for x in range(self._num_players)]
        return [0.0] * self._num_players

    def observation_tensor(self, player):
        # obs = [-1 if x < self._num_players else 0 for x in range(10)]
        obs = [0 for x in range(10)]
        obs[player] = 1

        obs.extend([1 if x < self._num_players else 0 for x in range(10)])
        obs[10+player] = 0

        for x in range(4):
            obs.extend([0 for y in range(104)])
            for card in self._boards[x]:
                obs[-105 + card] = 1

        obs.extend([0 for y in range(104)])
        for card in self._hands[player]:
            obs[-105 + card] = 1

        for x in range(10):
            obs.extend([0 for y in range(104)])
            if x < self._num_players:
                for card in self._bulls[x]:
                    obs[-105 + card] = 1

        for x in range(10):
            obs.extend([0 for y in range(104)])
            if self._mode == 1 and x < self._num_players:
                for card, splayer in self._stacks:
                    if x == splayer:
                        obs[-105 + card] = 1

        return np.array(obs)

    def full_observation_tensor(self, player):
        # obs = [-1 if x < self._num_players else 0 for x in range(10)]
        obs = [0 for x in range(10)]
        obs[player] = 1

        obs.extend([1 if x < self._num_players else 0 for x in range(10)])
        obs[10+player] = 0

        for x in range(4):
            obs.extend([0 for y in range(104)])
            for card in self._boards[x]:
                obs[-105 + card] = 1

        obs.extend([0 for y in range(104)])
        for card in self._hands[player]:
            obs[-105 + card] = 1

        for x in range(10):
            obs.extend([0 for y in range(104)])
            if x < self._num_players:
                for card in self._bulls[x]:
                    obs[-105 + card] = 1

        for x in range(10):
            obs.extend([0 for y in range(104)])
            if x < self._num_players:
                for card, splayer in self._stacks:
                    if x == splayer:
                        obs[-105 + card] = 1

        return np.array(obs)

    def legal_actions(self):
        if self.is_terminal():
            return []

        if self._mode == 0:
            return list(self._hands[self._cur_player])

        return [105, 106, 107, 108]

    def legal_actions_mask(self, player):
        action_mask = [0] * 109
        if self.is_terminal() or player != self._cur_player:
            return action_mask
        
        for action in self.legal_actions():
            action_mask[action] = 1
        return action_mask

    def apply_action(self, action):
        # print("apply action", self._cur_player, action)
        if self._mode == 0 and action < 105:
            self._hands[self._cur_player].discard(action)
            self._stacks.append((action, self._cur_player))
            self._cur_player += 1

            if self._cur_player == self._num_players:
                self._mode = 1
                self._stack_idx = 0
                self._stacks.sort()
                self.place_card_from_stack()
        elif self._mode == 1 and action > 104 and action < 109:
            card, player = self._stacks[self._stack_idx]
            self.replace_row(action-105, card, player)
            self._stack_idx += 1
            self.place_card_from_stack()
        else:
            raise Exception("Invalid move {} {}".format(self._mode, action))

    def action_to_string(self, player, action):
        if action < 105:
            return "(hand,{},{})".format(player, action)
        return "(stack,{},{})".format(player, action-104)

    def board_for_card(self, card):
        best_idx, best_val = -1, -1
        for row in range(4):
            if self._boards[row][-1] < card and (best_idx == -1 or self._boards[best_idx][-1] < self._boards[row][-1]):
                best_idx, best_val = row, self._boards[row][-1]

        return best_idx

    def replace_row(self, row, card, player):
        self._bulls[player].extend(self._boards[row])
        self._boards[row] = [card]

    def place_card_from_stack(self):
        while self._stack_idx < self._num_players:
            card, player = self._stacks[self._stack_idx]
            row = self.board_for_card(card)

            if row == -1:
                self._cur_player = player
                return
            if len(self._boards[row]) == 5:
                self.replace_row(row, card, player)
            else:
                self._boards[row].append(card)
            self._stack_idx += 1

        if len(self._hands[0]) > 0:
            self._mode = 0
            self._cur_player = 0
            self._stacks = []
        else:
            self._is_terminal = True

    def stack_str(self):
        if self._mode == 0:
            return ", ".join(map(str,self._stacks[0:self._cur_player]))
        else:
            return ", ".join(map(str,self._stacks[self._stack_idx:self._num_players]))

    def __str__(self):
        return "\n".join([
            "Boards",
            "\n".join(", ".join(map(str, board)) for board in self._boards),
            "Stacks",
            self.stack_str(),
            "Hands",
            "\n".join("{}: {}".format(player, ", ".join(map(str,hands))) for player, hands in enumerate(self._hands)),
            "Bulls",
            "\n".join("{} ({}): {}".format(player, str(bull_value(bull)), ", ".join(map(str,bull))) for player, bull in enumerate(self._bulls)),
        ])

    def clone(self):
        # return copy.deepcopy(self)
        a_copy = empty_copy(self)
        a_copy._cur_player = self._cur_player
        a_copy._mode = self._mode
        a_copy._num_players = self._num_players
        
        a_copy._hands = [set(hand) for hand in self._hands]
        a_copy._boards = [board[:] for board in self._boards]
        a_copy._bulls = [bull[:] for bull in self._bulls]
        a_copy._stacks = [stack[:] for stack in self._stacks]
        a_copy._stack_idx = self._stack_idx

        a_copy._is_terminal = self._is_terminal
        return a_copy


# new_initial_state
# num_distinct_actions
# num_players
# observation_tensor_shape

class NimmtGame(object):
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
        return NimmtState(self)

    def num_distinct_actions(self):
        return 109

    def num_players(self):
        return 10

    def observation_tensor_shape(self):
        return np.array([2620])
