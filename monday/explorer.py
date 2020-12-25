from absl import app

from az_model import config_to_model
from game import get_game
from play_game import play_and_explain
import utils.spawn as spawn
from utils.watcher import watcher
from az_config import az_config
import utils.logger as file_logger
import az_eval as evaluator_lib
import az_model as model_lib

def play_once(logger, config, game, model, game_num):
    az_evaluator = evaluator_lib.AlphaZeroEvaluator(game, model)
    evaluators = [az_evaluator.evaluate for player in range(game.num_players())]
    prior_fns = [az_evaluator.prior for player in range(game.num_players())]

    with file_logger.FileLogger(config.path + '/log', 'preview_' + str(game_num), config.quiet) as plogger:
        play_and_explain(plogger, game, evaluators, prior_fns)

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
    game = get_game(config.game)

    # Extend config with game data
    config = config._replace(
        observation_shape=game.observation_tensor_shape(),
        output_size=game.num_distinct_actions())
    model = config_to_model(config)
    if config.cp_num and config.path:
        model.load_checkpoint(config.path + '/cp/checkpoint-' + str(config.cp_num))
    
    play_once(logger, config, game, model, -1)

    # state = game.new_initial_state()
    # for x in range(1):
    #     print("Game", x)

    #     data = play_once(logger, config, game, model, -1)
    #     az_evaluator.clear_cache()

    #     trainInputs = []
    #     for d in data:
    #         for s in d.states:
    #             if all([x == 1 for x in s.legals_mask]):
    #                 state = game.new_initial_state()

    #             logger.print(az_evaluator._inference(state, state.current_player()))
    #             logger.print("Data:")
    #             logger.print(s.observation)
    #             logger.print(s.legals_mask)
    #             logger.print(s.policy)
    #             logger.print(d.returns[s.current_player], "\n\n")

    #             state.apply_action(s.action)

    #         trainInputs.extend([model_lib.TrainInput(s.observation, s.legals_mask, s.policy, d.returns[s.current_player]) for s in d.states])

    #     losses = model.update(trainInputs)
    #     logger.print(losses, "\n\n\n")

def main(unused_argv):
    simulate_training(config=az_config)

if __name__ == "__main__":
    with spawn.main_handler():
        app.run(main)
