class Evaluator(object):
  """Abstract class representing an evaluation function for a game.

  The evaluation function takes in an intermediate state in the game and returns
  an evaluation of that state, which should correlate with chances of winning
  the game. It returns the evaluation from all player's perspectives.
  """

  def evaluate(self, state):
    """Returns evaluation on given state."""
    raise NotImplementedError

  def prior(self, state):
    """Returns a probability for each legal action in the given state."""
    raise NotImplementedError


