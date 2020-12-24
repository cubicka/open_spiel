import tictactoe

def get_game(name):
    if (name == 'tictactoe'): return tictactoe.TicTacToeGame()
    raise Exception('Game {} is not found'.format(name))
