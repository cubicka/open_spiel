from absl import app
from config import az_config
from game import get_game
import utils.spawn as spawn
from model import MuModel
from selfplay import explore

def play_explore(config):
    game, config = get_game(config)
    config = config._replace(cp_num=1)
    model = MuModel(game.num_states(), game.num_actions())

    if config.cp_num and config.path:
        model.load_checkpoint(config.path + '/cp/checkpoint-' + str(config.cp_num))

    explore(config.path, model, game.clone(), config.cp_num)

def main(unused_argv):
    play_explore(config=az_config)

if __name__ == "__main__":
    with spawn.main_handler():
        app.run(main)
