from absl import app

from az_model import config_to_model
from game import get_game
from play_game import play_and_explain, play_and_explore
import utils.spawn as spawn
from utils.watcher import watcher
from az_config import az_config
import utils.logger as file_logger
import az_eval as evaluator_lib
import az_model as model_lib
from mcts.eval import mcts_evaluation, mcts_prior

def play_once(logger, config, game, model, game_num):
    az_evaluator = evaluator_lib.AlphaZeroEvaluator(model)
    # evaluators = [az_evaluator.evaluate for player in range(game.num_players())]
    # prior_fns = [az_evaluator.prior for player in range(game.num_players())]

    # evaluators_random = [az_evaluator.evaluate for player in range(game.num_players())]
    # prior_fns_random = [az_evaluator.prior for player in range(game.num_players())]

    # evaluators_mcts = [mcts_evaluation for player in range(game.num_players())]
    # evaluators_mcts[0] = az_evaluator.evaluate
    # prior_fns_mcts = [mcts_prior for player in range(game.num_players())]
    # prior_fns_mcts[0] = az_evaluator.prior

    with file_logger.FileLogger(config.path + '/log', 'preview_' + str(game_num), config.quiet) as plogger:
        # play_and_explore(game, evaluators, prior_fns, None)
        # play_and_explore(game, evaluators_mcts, prior_fns_mcts, None)

        play_and_explain(plogger, az_evaluator, game)

        # play_and_explain(plogger, game, state, evaluators, prior_fns, True, True)
        # play_and_explain(plogger, game, state, evaluators_mcts, prior_fns_mcts, True)
        # play_and_explain(plogger, game, state, evaluators, prior_fns)
        # play_and_explain(plogger, game, state, evaluators_mcts, prior_fns_mcts)

    # trajectories = []
    # with file_logger.FileLogger(config.path + '/log', 'preview_' + str(game_num), config.quiet) as plogger:
    #     az_evaluator = evaluator_lib.AlphaZeroEvaluator(game, model)

    #     evaluators = [az_evaluator.evaluate for player in range(game.num_players())]
    #     prior_fns = [az_evaluator.prior for player in range(game.num_players())]

    #     # for _ in range(game.num_players()):
    #     trajectories.append(play(plogger, game, evaluators, prior_fns, True))

    # return trajectories

@watcher
def simulate_training(config, logger):
    game, config = get_game(config)
    config = config._replace(cp_num=-1)
    model = config_to_model(config)
    if config.cp_num and config.path:
        model.load_checkpoint(config.path + '/cp/checkpoint-' + str(config.cp_num))
    
    az_evaluator = evaluator_lib.AlphaZeroEvaluator(game, model)

    play_once(logger, config, game, model, -1)
    # state = game.new_initial_state()
    # print(az_evaluator._inference(state, state.current_player()))

def main(unused_argv):
    simulate_training(config=az_config)

if __name__ == "__main__":
    with spawn.main_handler():
        app.run(main)
