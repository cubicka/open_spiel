from absl import app
from config import az_config
from game import get_game
import utils.spawn as spawn
from model import MuModel
from selfplay import explore
from diagnose import DiagnoseModel

def prevn(n):
    if n > 0: return n - 1
    return 4

def play_explore(config):
    game, config = get_game(config)
    model = MuModel(game.num_states(), game.num_actions(), s_dim=game.num_states())
    model.load_checkpoint(config.path + '/cp/checkpoint-' + str(config.cp_num))

    prevmodel = MuModel(game.num_states(), game.num_actions(), s_dim=game.num_states())
    prevmodel.load_checkpoint(config.path + '/cp/checkpoint-' + str(prevn(config.cp_num)))

    explore(config.path, prevmodel, model, game.clone(), config.cp_num)

def diagnose(config):
    game, config = get_game(config)
    model = MuModel(game.num_states(), game.num_actions(), s_dim=game.num_states())
    model.load_checkpoint(config.path + '/cp/checkpoint-' + str(config.cp_num))

    dm = DiagnoseModel(config, model)
    dm.diagnose(game)
    input("Press enter to close all plots")
    dm.close_all()

def main(unused_argv):
    config = az_config
    config = config._replace(cp_num=1)
    # diagnose(config)
    play_explore(config=config)

if __name__ == "__main__":
    with spawn.main_handler():
        app.run(main)
