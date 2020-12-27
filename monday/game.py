import tictactoe
import nimmt

def get_game(name):
    if (name == 'nimmt'): return nimmt.NimmtGame()
    if (name == 'tictactoe'): return tictactoe.TicTacToeGame()
    raise Exception('Game {} is not found'.format(name))
