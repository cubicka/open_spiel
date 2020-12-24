from absl import app

from az_model import config_to_model
from game import get_game
from play_game import play
import utils.spawn as spawn
from utils.watcher import watcher
from az_config import az_config
import utils.logger as file_logger
import az_eval as evaluator_lib
import az_model as model_lib

def play_once(logger, config, game, model, game_num):
    trajectories = []
    with file_logger.FileLogger(config.path + '/log', 'preview_' + str(game_num), config.quiet) as plogger:
        az_evaluator = evaluator_lib.AlphaZeroEvaluator(game, model)

        evaluators = [az_evaluator.evaluate for player in range(game.num_players())]
        prior_fns = [az_evaluator.prior for player in range(game.num_players())]

        for _ in range(game.num_players()):
            trajectories.append(play(plogger, game, evaluators, prior_fns, True))

    return trajectories

@watcher
def simulate_training(config, logger):
    game = get_game(config.game)

    # Extend config with game data
    config = config._replace(
        observation_shape=game.observation_tensor_shape(),
        output_size=game.num_distinct_actions())
    model = config_to_model(config)

    az_evaluator = evaluator_lib.AlphaZeroEvaluator(game, model)
    for x in range(11):
        print("Game", x)
        if x % 1 == 0:
            az_evaluator.clear_cache()
            state = game.new_initial_state()
            logger.print(az_evaluator._inference(state))

            state.apply_action(0)
            logger.print(az_evaluator._inference(state))

            cstate = state.clone()
            cstate.apply_action(7)
            logger.print(az_evaluator._inference(cstate))

            cstate = state.clone()
            cstate.apply_action(4)
            logger.print(az_evaluator._inference(cstate), "\n\n\n")

        data = play_once(logger, config, game, model, x%7)

        trainInputs = []
        for d in data:
            trainInputs.extend([model_lib.TrainInput(s.observation, s.legals_mask, s.policy, d.returns[s.current_player]) for s in d.states])

        losses = model.update(trainInputs)
        logger.print(losses)

def main(unused_argv):
    simulate_training(config=az_config)

if __name__ == "__main__":
    with spawn.main_handler():
        app.run(main)
