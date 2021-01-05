import numpy as np
import copy
from utils.empty_copy import empty_copy
from numba import jit
from numba.typed import List
# from numba.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
# import warnings

# warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
# warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

@jit(nopython=True)
def card_value(card):
    if (card == 55): return 7
    if (card%11 == 0): return 5
    if (card%10 == 0): return 3
    if (card%5 == 0): return 2
    return 1

@jit(nopython=True)
def bull_value(cards):
    return np.sum(np.array([card_value(idx[0]+1) for idx, card in np.ndenumerate(cards) if card != 0]))
    # sum = 0
    # for idx, card in np.ndenumerate(cards):
    #     if card != 0:
    #         sum += card_value(idx[0]+1)
    # return sum

@jit(nopython=True)
def bulls_scores(bulls, n_players, is_terminal):
    scores = np.zeros(10)
    if not is_terminal:
        return scores

    values = [bull_value(bull) for bull in bulls]
    best_value = min(values)
    worst_value = max(values)
    if best_value == worst_value:
        return scores

    n_best = len([v for v in values if v == best_value])
    n_worst = len([v for v in values if v == worst_value])
    for x in range(n_players):
        if values[x] == best_value:
            scores[x] = 1.0 / n_best
        elif values[x] == worst_value:
            scores[x] = -1.0 / n_worst
    return scores

@jit(nopython=True)
def initial_state(n_player):
    shuffled_cards = np.arange(1, 105)
    np.random.shuffle(shuffled_cards)

    # print([shuffled_cards[x*10:x*10+10].copy() for x in range(n_player)])
    hands = shuffled_cards[0:n_player*10].reshape((n_player, 10)).copy()
    # boards = np.array([shuffled_cards[n_player*10+x:n_player*10+x+1] for x in range(4)])
    boards = np.zeros((4,5), dtype=np.int32)
    for x in range(4):
        boards[x][0] = shuffled_cards[n_player*10+x]
    bulls = np.zeros((n_player, 104), dtype=np.int32)
    stacks = np.empty((n_player, 2), dtype=np.int32)
    return hands, boards, bulls, stacks

@jit(nopython=True)
def observation_tensor(mode, num_players, player, hands, boards, bulls, stacks):
    # print(mode, num_players, player, hands, boards, bulls, stacks)
    obs = np.zeros(2620, dtype=np.int32)
    obs[player] = 1
    sx = 10

    for x in range(num_players):
        obs[sx+x] = 1
    obs[sx+player] = 0
    sx += 10

    for card in hands[player][hands[player] != 0]:
        obs[sx+card-1] = 1
    sx += 104

    for board in boards:
        for card in board[board != 0]:
            obs[sx+card-1] = 1
        sx += 104

    for x in range(10):
        if x < num_players:
            for idx, card in np.ndenumerate(bulls[x]):
                if card != 0: obs[sx+idx[0]] = 1
        sx += 104

    if mode == 1:
        for card, splayer in stacks:
            obs[sx + splayer*104 + card] = 1

    return obs

@jit(nopython=True)
def legal_actions(mode, hands, player):
    if mode == 1: return np.array([104, 105, 106, 107])
    return hands[player][hands[player] != 0] - 1

@jit(nopython=True)
def row_idx_of_card(boards, card):
    best_idx, best_val = -1, -1
    for row in range(4):
        row_val = max(boards[row])
        if row_val < card and (best_idx == -1 or best_val < row_val):
            best_idx, best_val = row, row_val
    return best_idx

@jit(nopython=True)
def replace_row(boards, bulls, row, card, player):
    for c in boards[row][boards[row] != 0]:
        bulls[player][c-1] = 1
    # bulls[player].extend(boards[row])
    # print("replaced-before", boards, card, row)
    boards[row] = [card, 0, 0, 0, 0]
    # print("replaced", boards)

@jit(nopython=True)
def place_card_from_stack(nplayer, boards, hands, bulls, stacks, stack_idx):
    while stack_idx < nplayer:
        card, player = stacks[stack_idx]
        row = row_idx_of_card(boards, card)

        if row == -1:
            return 1, player, stack_idx, False, stacks

        if np.all(boards[row] != 0):
            # print("place card")
            replace_row(boards, bulls, row, card, player)
        else:
            for x in range(5):
                if boards[row][x] == 0:
                    boards[row][x] = card
                    break
            # print("come on", boards[row])
        stack_idx += 1

    if np.any(hands[0] != 0):
        return 0, 0, 0, False, np.empty((nplayer, 2), dtype=np.int32)
    else:
        return 0, 0, 0, True, np.empty((nplayer, 2), dtype=np.int32)

@jit(nopython=True)
def apply_action(action, nplayer, mode, player, stack_idx, boards, hands, bulls, stacks):
    # print("action", player, action)
    if mode == 0 and action < 105:
        for x in range(10):
            if hands[player][x] == action + 1:
                hands[player][x] = 0
                break
        # hands[player] = hands[player][hands[player] != action]
        stacks[player] = (action + 1, player)
        player += 1

        if player == nplayer:
            stacks = stacks[stacks[:,0].argsort()]
            # print("stacks", stacks)
            return place_card_from_stack(nplayer, boards, hands, bulls, stacks, 0)
    elif mode == 1 and action > 103 and action < 108:
        card, _ = stacks[stack_idx]
        # print("pre 1", boards)
        # print("stacks!", stacks, player, action, card)
        replace_row(boards, bulls, action-104, card, player)
        # print("pre 1", boards)
        return place_card_from_stack(nplayer, boards, hands, bulls, stacks, stack_idx + 1)
    # else: raise Exception("Invalid move {} {}".format(mode, action))

    return 0, player, 0, False, stacks


class Nimmt(object):
    def __init__(self):
        self._cur_player = 0
        self._mode = 0
        self._num_players = np.random.randint(2, 11)

        # self._hands = List([set([0])])
        # self._bulls = List([List([0])])
        # self._boards = List([List([0])])
        # self._stacks = List([(1,1)])
        self._stack_idx = 0
        self._is_terminal = False

        self._hands, self._boards, self._bulls, self._stacks = initial_state(self._num_players)

    def clone(self):
        return copy.deepcopy(self)
        # a_copy = empty_copy(self)
        # a_copy._cur_player = self._cur_player
        # a_copy._mode = self._mode
        # a_copy._num_players = self._num_players

        # a_copy._hands = [set(hands) for hands in self._hands]
        # a_copy._boards = [boards[:] for boards in self._boards]
        # a_copy._bulls = [bulls[:] for bulls in self._bulls]
        # a_copy._stacks = self._stacks[:]
        # a_copy._stack_idx = self._stack_idx

        # a_copy._is_terminal = self._is_terminal

        # return a_copy

    def reset(self):
        self.__init__()

    def num_players(self):
        return 10

    def num_actions(self):
        return 108

    def num_states(self):
        return 2620

    def is_simultaneous_node(self):
        return self._mode == 0 and self._cur_player < self._num_players - 1

    def is_terminal(self):
        return self._is_terminal

    def current_player(self):
        return -1 if self._is_terminal else self._cur_player

    def returns(self):
        return bulls_scores(self._bulls, self._num_players, self._is_terminal)

    def observation_tensor(self):
        # print(self._mode, self._num_players, self._cur_player, self._boards, self._hands, self._bulls, self._stacks)
        return observation_tensor(self._mode, self._num_players, self._cur_player, self._hands, self._boards, self._bulls, self._stacks)

    def legal_actions(self):
        if self._is_terminal: return np.array([])
        return legal_actions(self._mode, self._hands, self._cur_player)

    def legal_actions_mask(self):
        action_mask = [0] * 108
        for action in self.legal_actions():
            action_mask[action] = 1
        return action_mask

    def apply_action(self, action):
        self._mode, self._cur_player, self._stack_idx, self._is_terminal, self._stacks = apply_action(action, self._num_players, self._mode, self._cur_player, self._stack_idx, self._boards, self._hands, self._bulls, self._stacks)

    def action_to_string(self, player, action):
        if action < 104:
            return "(h,{},{})".format(player, action+1)
        return "(s,{},{})".format(player, action-103)

    def stack_str(self):
        if self._mode == 0:
            return ", ".join(map(str,self._stacks[0:self._cur_player]))
        else:
            return ", ".join(map(str,self._stacks[self._stack_idx:self._num_players]))

    def __str__(self):
        return "\n".join([
            "Boards",
            "\n".join(", ".join(map(lambda x: str(x), board[board!=0])) for board in self._boards),
            "Stacks",
            self.stack_str(),
            "Hands",
            "\n".join("{}: {}".format(player, ", ".join(map(lambda x: str(x),hands[hands != 0]))) for player, hands in enumerate(self._hands)),
            "Bulls",
            "\n".join("{} ({}): {}".format(player, str(bull_value(bull)), ", ".join([str(idx[0]+1) for idx, x in np.ndenumerate(bull) if x != 0])) for player, bull in enumerate(self._bulls)),
        ])
