from absl import app

from az_config import az_config
import az_eval as evaluator_lib
from az_model import config_to_model
from game import get_game
import utils.logger as file_logger

def play_game(game):
    state = game.new_initial_state()
    file = open("./monday/in.txt", "r") 

    with file_logger.FileLogger('./ma', 'out', True) as logger:
        lines = file.readlines()
        players = lines[0].split()
        nplayer = len(players)
        logger.print(nplayer, players)
        state._num_players = nplayer
        state._bulls = [[] for _ in range(state._num_players)]
        state._hands = [set() for _ in range(nplayer)]

        boards = list(map(int, lines[1].split()))
        for idx, card in enumerate(boards):
            state._boards[idx] = [card]

        hands = list(map(int, lines[2].split()))
        for card in hands:
            state._hands[0].add(card)

        for line in lines[3:]:
            if line.startswith('c'):
                player_actions = {}
                s = line.split()[1:]
                for idx in range(0, len(s), 2):
                    player_actions[s[idx]] = int(s[idx+1])
                logger.print(player_actions)
                for x in range(nplayer):
                    state.apply_action(player_actions[players[x]])
            elif line.startswith('s'):
                state.apply_action(int(line.split()[1]) + 104)
            else:
                raise Exception('invalid command!')

        priors = az_evaluator.prior(state)
        if len(priors) > 0:
            best_action_prior = max(priors, key=lambda probs: probs[1])
            best_action = best_action_prior[0]
            logger.print(best_action, az_evaluator.evaluate(state, 0))
        else:
            logger.print(state.returns())
        logger.print('state', state)

game_name = 'nimmt'
cp = -1
game = get_game('nimmt')
config = az_config._replace(
    observation_shape=game.observation_tensor_shape(),
    output_size=game.num_distinct_actions())
model = config_to_model(config)
model.load_checkpoint(config.path + '/cp/checkpoint--1')
az_evaluator = evaluator_lib.AlphaZeroEvaluator(game, model)

while True:
    s = input("press enter:")
    play_game(game)
