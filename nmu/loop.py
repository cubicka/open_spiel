from absl import app
import collections
import datetime
import functools
import itertools
import json
import os
import random
import sys
import tempfile
import time
import traceback
from tqdm import tqdm

import numpy as np

# import mcts
# import az_eval as evaluator_lib
# import az_model as model_lib
from config import az_config

import tictactoe
# import nimmt

import utils.logger as file_logger
import utils.spawn as spawn
from utils.watcher import watcher
# from mcts_bot import mcts_search
# from mcts.eval import mcts_evaluation, mcts_prior
from trajectory import Trajectory, TrajectoryState
from game import get_game
# from explorer import play_once
from selfplay import play, explore, history_to_target
from model import MuModel, Losses

class Buffer(object):
  """A fixed size buffer that keeps the newest values."""

  def __init__(self, max_size):
    self.max_size = max_size
    self.data = []
    self.total_seen = 0  # The number of items that have passed through.

  def __len__(self):
    return len(self.data)

  def __bool__(self):
    return bool(self.data)

  def append(self, val):
    return self.extend([val])

  def extend(self, batch):
    batch = list(batch)
    self.total_seen += len(batch)
    self.data.extend(batch)
    self.data[:-self.max_size] = []

  def sample(self, count):
    return random.sample(self.data, count)


def update_checkpoint(logger, queue, model):
  """Read the queue for a checkpoint to load, or an exit signal."""
  path = None
  while True:  # Get the last message, ignore intermediate ones.
    try:
      path = queue.get_nowait()
    except spawn.Empty:
      break
  if path:
    # logger.print("Inference cache:", az_evaluator.cache_info())
    logger.print("Loading checkpoint", path)
    model.load_checkpoint(path)
    # model.clear_cache()
  elif path is not None:  # Empty string means stop this process.
    return False
  return True


@watcher
def actor(*, config, game, logger, queue):
    """An actor process runner that generates games and returns trajectories."""
    logger.print("Initializing model")
    # model = model_lib.config_to_model(config)
    model = MuModel(game.num_states(), game.num_actions())

    logger.print("Initializing bots")
    # az_evaluator = evaluator_lib.AlphaZeroEvaluator(model)

    for game_num in itertools.count():
        if not update_checkpoint(logger, queue, model):
            return

        # print("wait actor", game_num)
        game.reset()
        queue.put(play(model, game.clone()))


@watcher
def learner(*, game, config, actors, broadcast_fn, logger):
    """A learner that consumes the replay buffer and trains the network."""
    logger.print("Initializing model")
    # model = model_lib.config_to_model(config)
    model = MuModel(game.num_states(), game.num_actions())
    logger.print("Model size:", model.num_trainable_variables, "variables")
    if config.cp_num and config.path:
        model.load_checkpoint(config.path + '/cp/checkpoint-' + str(config.cp_num))
  
    save_path = model.save_checkpoint(config.path + '/cp/checkpoint-' + str(config.cp_num))
    broadcast_fn(save_path)
    logger.print("Initial checkpoint:", save_path)

    replay_buffer = Buffer(config.replay_buffer_size)
    learn_rate = config.replay_buffer_size // config.replay_buffer_reuse
    total_trajectories = 0

    explore(config.path, model, game.clone(), 0)
    a_dim = game.num_actions()

    def trajectory_generator():
        """Merge all the actor queues into a single generator."""
        while True:
            found = 0
            for actor_process in actors:
                try:
                    yield actor_process.queue.get_nowait()
                    # print("got results back")
                except spawn.Empty:
                    pass
                else:
                    found += 1
            if found == 0:
                time.sleep(1)  # 10ms

    def collect_trajectories():
        """Collects the trajectories from actors into the replay buffer."""
        num_trajectories = 0
        num_states = 0
        # replay_buffer.data = []
        # print("Collection trajectories")
        with tqdm(total=learn_rate) as pbar:
          for trajectory in trajectory_generator():
            targets = history_to_target(a_dim, trajectory)
            # print(trajectory)
            # print("collect", len(targets))
            num_trajectories += 1
            num_states += len(targets)
            pbar.update(len(targets))

            replay_buffer.extend(targets)

            # print("Getting trajectories {}/{}".format(num_states, learn_rate))

            # if num_trajectories > 10:
            if num_states >= learn_rate:
                break

        # print("Returning trajectories", num_trajectories, num_states)
        return num_trajectories, num_states

    def learn(step):
        """Sample from the replay buffer, update weights and save a checkpoint."""
        losses = []
        # print("Start learning #{}".format(step))
        for epoch in range(len(replay_buffer) // config.train_batch_size):
            data = replay_buffer.sample(config.train_batch_size)
            obs, acts0, acts1, acts2, acts3, pols0, pols1, pols2, pols3, pols4, rets0, rets1, rets2, rets3, rets4 = zip(*data)
            losses.append(model.train(obs, acts0, acts1, acts2, acts3, pols0, pols1, pols2, pols3, pols4, rets0, rets1, rets2, rets3, rets4))
            # print("Learn Epoch #{}".format(epoch))
            # print(losses[len(losses) - 1])

        # Always save a checkpoint, either for keeping or for loading the weights to
        # the actors. It only allows numbers, so use -1 as "latest".
        # save_path = model.save_checkpoint(
        #     step if step % config.checkpoint_freq == 0 else -1)
        save_path = model.save_checkpoint(config.path + '/cp/checkpoint-' + str(step % 5))
        logger.print(len(replay_buffer), config.train_batch_size, len(losses))
        losses = sum(losses, Losses(0, 0, 0)) / len(losses)
        logger.print(losses)
        logger.print("Checkpoint saved:", save_path)
        print("#{}".format(step), losses)
        # logger.print("Sample data", replay_buffer.sample(1))

        explore(config.path, model, game.clone(), step % 5)
        return save_path, losses

    last_time = time.time() - 60
    prevl2 = None
    prevl2Count = 0
    for step in itertools.count(1):
        num_trajectories, num_states = collect_trajectories()
        total_trajectories += num_trajectories
        now = time.time()
        seconds = now - last_time
        last_time = now

        logger.print("Step:", step)
        logger.print(
            ("Collected {:5} states from {:3} games, {:.1f} states/s. "
            "{:.1f} states/(s*actor), game length: {:.1f}").format(
                num_states, num_trajectories, num_states / seconds,
                num_states / (config.actors * seconds),
                num_states / num_trajectories))
        logger.print("Buffer size: {}. States seen: {}".format(
            len(replay_buffer), replay_buffer.total_seen))

        save_path, losses = learn(step)
        broadcast_fn(save_path)
        logger.print()

        latest_losses = losses[len(losses) - 1]
        if latest_losses == prevl2:
          prevl2Count += 1
        else:
          prevl2Count = 0
          prevl2 = latest_losses

        if (config.max_steps > 0 and step >= config.max_steps) or (latest_losses < config.learning_rate) or prevl2Count >= 10:
            break

def alpha_zero(config):
    """Start all the worker processes for a full alphazero setup."""
    game, config = get_game(config)

    print("Starting game", config.game)

    path = config.path
    if not path:
      path = tempfile.mkdtemp(prefix="az-{}-{}-".format(
          datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"), config.game))
      config = config._replace(path=path)

    if not os.path.exists(path):
      os.makedirs(path)
      os.makedirs(path + '/log')
      os.makedirs(path + '/cp')

    if not os.path.isdir(path):
      sys.exit("{} isn't a directory".format(path))

    print("Writing logs and checkpoints to:", path)
    print("Model type: %s(%s, %s)" % (config.nn_model, config.nn_width,
                                      config.nn_depth))

    actors = [spawn.Process(actor, kwargs={"game": game.clone(), "config": config,
                                            "num": i})
                for i in range(config.actors)]

    def broadcast(msg):
        for proc in actors:
            proc.queue.put(msg)

    try:
      learner(game=game.clone(), config=config, actors=actors, broadcast_fn=broadcast, num=config.nn_depth)
    except (KeyboardInterrupt, EOFError):
      print("Caught a KeyboardInterrupt, stopping early.")
    finally:
      broadcast("")
      # for actor processes to join we have to make sure that their q_in is empty,
      # including backed up items
      for proc in actors:
        while proc.exitcode is None:
          while not proc.queue.empty():
            proc.queue.get_nowait()
          proc.join(0.001)
        for proc in evaluators:
          proc.join()

def main(unused_argv):
    alpha_zero(config=az_config)

if __name__ == "__main__":
    with spawn.main_handler():
        app.run(main)
