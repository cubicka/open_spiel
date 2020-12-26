# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""A basic AlphaZero implementation.

This implements the AlphaZero training algorithm. It spawns N actors which feed
trajectories into a replay buffer which are consumed by a learner. The learner
generates new weights, saves a checkpoint, and tells the actors to update. There
are also M evaluators running games continuously against a standard MCTS+Solver,
though each at a different difficulty (ie number of simulations for MCTS).

Due to the multi-process nature of this algorithm the logs are written to files,
one per process. The learner logs are also output to stdout. The checkpoints are
also written to the same directory.

Links to relevant articles/papers:
  https://deepmind.com/blog/article/alphago-zero-starting-scratch has an open
    access link to the AlphaGo Zero nature paper.
  https://deepmind.com/blog/article/alphazero-shedding-new-light-grand-games-chess-shogi-and-go
    has an open access link to the AlphaZero science paper.
"""

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

import numpy as np

import mcts
import az_eval as evaluator_lib
import az_model as model_lib
from az_config import az_config

import tictactoe
import nimmt

import utils.logger as file_logger
import utils.spawn as spawn
from utils.watcher import watcher
from mcts_bot import mcts_search
from mcts.eval import mcts_evaluation, mcts_prior
from trajectory import Trajectory, TrajectoryState
from game import get_game
from explorer import play_once
from play_game import play_and_explore

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


def update_checkpoint(logger, queue, model, az_evaluator):
  """Read the queue for a checkpoint to load, or an exit signal."""
  path = None
  while True:  # Get the last message, ignore intermediate ones.
    try:
      path = queue.get_nowait()
    except spawn.Empty:
      break
  if path:
    logger.print("Inference cache:", az_evaluator.cache_info())
    logger.print("Loading checkpoint", path)
    model.load_checkpoint(path)
    az_evaluator.clear_cache()
  elif path is not None:  # Empty string means stop this process.
    return False
  return True


@watcher
def actor(*, config, game, logger, queue):
    """An actor process runner that generates games and returns trajectories."""
    logger.print("Initializing model")
    model = model_lib.config_to_model(config)

    logger.print("Initializing bots")

    az_evaluator = evaluator_lib.AlphaZeroEvaluator(game, model)
    evaluators = [az_evaluator.evaluate, az_evaluator.evaluate]
    prior_fns = [az_evaluator.prior, az_evaluator.prior]
    for game_num in itertools.count():
        if not update_checkpoint(logger, queue, model, az_evaluator):
            return

        queue.put(play_and_explore(game, evaluators, prior_fns, (az_evaluator.reuse_policy, az_evaluator.save_policy)))


@watcher
def learner(*, game, config, actors, broadcast_fn, logger):
    """A learner that consumes the replay buffer and trains the network."""
    logger.print("Initializing model")
    model = model_lib.config_to_model(config)
    logger.print("Model size:", model.num_trainable_variables, "variables")
    if config.cp_num and config.path:
        model.load_checkpoint(config.path + '/cp/checkpoint-' + str(config.cp_num))
  
    save_path = model.save_checkpoint(0)
    broadcast_fn(save_path)
    logger.print("Initial checkpoint:", save_path)

    replay_buffer = Buffer(config.replay_buffer_size)
    learn_rate = config.replay_buffer_size // config.replay_buffer_reuse
    total_trajectories = 0

    play_once(logger, config, game, model, 0)

    def trajectory_generator():
        """Merge all the actor queues into a single generator."""
        while True:
            found = 0
            for actor_process in actors:
                try:
                    yield actor_process.queue.get_nowait()
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
        print("Collection trajectories")
        for trajectory in trajectory_generator():
          # for trajectory in trajectories:
          num_trajectories += 1
          num_states += len(trajectory.states)

          replay_buffer.extend(
              model_lib.TrainInput(
                  s.observation, s.legals_mask, s.policy, trajectory.returns[s.current_player])
              for s in trajectory.states)

          print("Getting trajectories {}/{}".format(num_states, learn_rate))

          if num_states >= learn_rate:
              break

        print("Returning trajectories", num_trajectories, num_states)
        return num_trajectories, num_states

    def learn(step):
        """Sample from the replay buffer, update weights and save a checkpoint."""
        losses = []
        print("Start learning #{}".format(step))
        for epoch in range(len(replay_buffer) // config.train_batch_size):
            data = replay_buffer.sample(config.train_batch_size)
            losses.append(model.update(data))
            print("Learn Epoch #{}".format(epoch))
            print(losses[len(losses) - 1])

        # Always save a checkpoint, either for keeping or for loading the weights to
        # the actors. It only allows numbers, so use -1 as "latest".
        save_path = model.save_checkpoint(
            step if step % config.checkpoint_freq == 0 else -1)
        losses = sum(losses, model_lib.Losses(0, 0, 0)) / len(losses)
        logger.print(losses)
        logger.print("Checkpoint saved:", save_path)

        play_once(logger, config, game, model, step % 5)
        return save_path, losses

    last_time = time.time() - 60
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

def alpha_zero(config):
    """Start all the worker processes for a full alphazero setup."""
    game = get_game(config.game)
    config = config._replace(
        observation_shape=game.observation_tensor_shape(),
        output_size=game.num_distinct_actions())

    print("Starting game", config.game)

    path = config.path
    if not path:
      path = tempfile.mkdtemp(prefix="az-{}-{}-".format(
          datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"), config.game))
      config = config._replace(path=path)

    if not os.path.exists(path):
      os.makedirs(path)
    if not os.path.isdir(path):
      sys.exit("{} isn't a directory".format(path))

    print("Writing logs and checkpoints to:", path)
    print("Model type: %s(%s, %s)" % (config.nn_model, config.nn_width,
                                      config.nn_depth))

    actors = [spawn.Process(actor, kwargs={"game": game, "config": config,
                                            "num": i})
                for i in range(config.actors)]

    def broadcast(msg):
        for proc in actors:
            proc.queue.put(msg)

    try:
      learner(game=game, config=config, actors=actors, broadcast_fn=broadcast)
    except (KeyboardInterrupt, EOFError):
      print("Caught a KeyboardInterrupt, stopping early.")
    finally:
      broadcast("")
      for proc in actors:
        proc.join()

def main(unused_argv):
    alpha_zero(config=az_config)

if __name__ == "__main__":
    with spawn.main_handler():
        app.run(main)
