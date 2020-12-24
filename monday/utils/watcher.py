import functools
import utils.logger as file_logger
import traceback

def watcher(fn):
  """A decorator to fn/processes that gives a logger and logs exceptions."""
  @functools.wraps(fn)
  def _watcher(*, config, num=None, **kwargs):
    """Wrap the decorated function."""
    name = fn.__name__
    if num is not None:
      name += "-" + str(num)
    with file_logger.FileLogger(config.path + '/log', name, config.quiet) as logger:
      print("{} started".format(name))
      logger.print("{} started".format(name))
      try:
        return fn(config=config, logger=logger, **kwargs)
      except Exception as e:
        logger.print("\n".join([
            "",
            " Exception caught ".center(60, "="),
            traceback.format_exc(),
            "=" * 60,
        ]))
        print("Exception caught in {}: {}".format(name, e))
        raise
      finally:
        logger.print("{} exiting".format(name))
        print("{} exiting".format(name))
  return _watcher

