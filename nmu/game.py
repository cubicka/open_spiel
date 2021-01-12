import tictactoe
# import nimmt

def get_game(config):
    # if (config.game == 'nimmt'): game = nimmt.Nimmt()
    if (config.game == 'tictactoe'): game = tictactoe.TicTacToe()
    else: raise Exception('Game {} is not found'.format(config.game))

    config = config._replace(
        observation_shape=game.num_states(),
        output_size=game.num_actions(),
        value_size=game.num_players(),
        path=config.path + '--' + str(config.nn_depth))

    return game, config
