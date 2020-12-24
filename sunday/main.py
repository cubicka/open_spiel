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

import tictactoe
import nimmt

import utils.logger as file_logger
import utils.spawn as spawn
from mcts_bot import mcts_search
from mcts.eval import mcts_evaluation, mcts_prior
from trajectory import Trajectory, TrajectoryState

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


class Config(collections.namedtuple(
    "Config", [
        "game",
        "cp_num",
        "path",
        "learning_rate",
        "weight_decay",
        "train_batch_size",
        "replay_buffer_size",
        "replay_buffer_reuse",
        "max_steps",
        "checkpoint_freq",
        "actors",
        "evaluators",
        "evaluation_window",
        "eval_levels",

        "uct_c",
        "max_simulations",
        "policy_alpha",
        "policy_epsilon",
        "temperature",
        "temperature_drop",

        "nn_model",
        "nn_width",
        "nn_depth",
        "observation_shape",
        "output_size",

        "quiet",
    ])):
  """A config for the model/experiment."""
  pass


def _init_model_from_config(config):
  return model_lib.Model.build_model(
      config.nn_model,
      config.observation_shape,
      config.output_size,
      config.nn_width,
      config.nn_depth,
      config.weight_decay,
      config.learning_rate,
      config.path)

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


def _init_bot(config, game, evaluator_, evaluation):
  """Initializes a bot."""
  noise = None if evaluation else (config.policy_epsilon, config.policy_alpha)
  return mcts.MCTSBot(
      game,
      config.uct_c,
      config.max_simulations,
      evaluator_,
      solve=True,
      dirichlet_noise=noise,
      child_selection_fn=mcts.SearchNode.puct_value,
      verbose=False)

def print_obs_tensor(logger, tensor):
  obs = []
  for i in range(len(tensor)):
    if tensor[i] != 0: obs.append(i)
  logger.print("Tensor:\n", [[x//104, x%104 if tensor[x] > 0 else -1*(x%104)] for x in obs])

def _play_game(logger, game_num, game, evaluators, prior_fns, temperature, temperature_drop, fprint):
  """Play one game, return the trajectory."""
  trajectory = Trajectory()
  actions = []
  state = game.new_initial_state()
#   logger.print("Initial obs", state.observation_tensor())
  if fprint:
    logger.print(" Starting game {} ".format(game_num).center(60, "-"))
    logger.print("Initial state:\n{}".format(state))

  is_prev_state_simultaneous = False
  while not state.is_terminal():
    # root = bots[state.current_player()].mcts_search(state)
    if is_prev_state_simultaneous:
        root = chosen_child
    else:
        root = mcts_search(evaluators, prior_fns, 2, temperature_drop != 0, state)

    policy = np.zeros(game.num_distinct_actions())
    root_solved = root.outcome is not None
    for c in root.children:
        if not root_solved:
            policy[c.action] = c.explore_count
        else:
            policy[c.action] = 1 if c.outcome is not None and c.outcome[state.current_player()] >= root.outcome[state.current_player()] else 0

    policy = policy ** (1 / temperature)
    policy /= policy.sum()        

    if len(actions) >= temperature_drop:
        action = root.best_child().action
    else:
        action = np.random.choice(len(policy), p=policy)
    chosen_child = filter(lambda c: c.action == action, root.children)

    trajectory.states.append(TrajectoryState(
        state.observation_tensor(), state.current_player(),
        state.legal_actions_mask(), action, policy,
        root.total_reward / root.explore_count))

    action_str = state.action_to_string(state.current_player(), action)
    actions.append(action_str)
    if fprint:
        logger.print("Root:")
        logger.print("\n", root.to_str(state))
        # print_obs_tensor(logger, state.observation_tensor())
        logger.print("Children:")
        logger.print("\n" + root.children_str(state))
        logger.print("======= Player {} sampled action: {}".format(
            state.current_player(), action_str, policy[action]))
        # logger.print("With policy {}".format(policy))
        logger.print("\n\n\n")

    is_prev_state_simultaneous = state.is_simultaneous_node()
    state.apply_action(action)
    if fprint:
        logger.print("Next state:\n{}".format(state))

#   logger.print("FInal obs", state.observation_tensor())
  trajectory.returns = state.returns()
  if fprint:
    logger.print("Game {}: Returns: {}; Actions: {}".format(
        game_num, " ".join(map(str, trajectory.returns)), " ".join(actions)))
  return trajectory


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
    model = _init_model_from_config(config)

    logger.print("Initializing bots")
    az_evaluator = evaluator_lib.AlphaZeroEvaluator(game, model)
    # bots = [_init_bot(config, game, az_evaluator, False) for _ in range(game.num_players())]
    evaluators = [az_evaluator.evaluate, az_evaluator.evaluate]
    prior_fns = [az_evaluator.prior, az_evaluator.prior]

    for game_num in itertools.count():
        if not update_checkpoint(logger, queue, model, az_evaluator):
            return
        queue.put(_play_game(logger, game_num, game, evaluators, prior_fns, config.temperature,
                                config.temperature_drop, False))


@watcher
def evaluator(*, game, config, logger, queue):
  """A process that plays the latest checkpoint vs standard MCTS."""
  results = Buffer(config.evaluation_window)
  logger.print("Initializing model")
  model = _init_model_from_config(config)
  logger.print("Initializing bots")
  az_evaluator = evaluator_lib.AlphaZeroEvaluator(game, model)
  random_evaluator = mcts.RandomRolloutEvaluator()

  for game_num in itertools.count():
    if not update_checkpoint(logger, queue, model, az_evaluator):
      return

    az_player = game_num % 2
    difficulty = (game_num // 2) % config.eval_levels
    max_simulations = int(config.max_simulations * (10 ** (difficulty / 2)))
    bots = [
        _init_bot(config, game, az_evaluator, True),
        mcts.MCTSBot(
            game,
            config.uct_c,
            max_simulations,
            random_evaluator,
            solve=True,
            verbose=False)
    ]
    if az_player == 1:
      bots = list(reversed(bots))

    trajectory = _play_game(logger, game_num, game, bots, temperature=1,
                            temperature_drop=0, fprint=False)
    results.append(trajectory.returns[az_player])
    queue.put((difficulty, trajectory.returns[az_player]))

    logger.print("AZ: {}, MCTS: {}, AZ avg/{}: {:.3f}".format(
        trajectory.returns[az_player],
        trajectory.returns[1 - az_player],
        len(results), np.mean(results.data)))


@watcher
def learner(*, game, config, actors, evaluators, broadcast_fn, logger):
    """A learner that consumes the replay buffer and trains the network."""
    logger.print("Initializing model")
    model = _init_model_from_config(config)
    # model.load_checkpoint('./cubick/cp/checkpoint-10')
    logger.print("Model size:", model.num_trainable_variables, "variables")
  
    save_path = model.save_checkpoint(0)
    broadcast_fn(save_path)
    logger.print("Initial checkpoint:", save_path)

    replay_buffer = Buffer(config.replay_buffer_size)
    learn_rate = config.replay_buffer_size // config.replay_buffer_reuse
    total_trajectories = 0

    simulate_model(logger, config, game, model, 0)

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

        # if step % config.checkpoint_freq == 0:
        simulate_model(logger, config, game, model, step if step % config.checkpoint_freq == 0 else -1)

        # one_sample = replay_buffer.sample(1)
        # logger.print("-------\nRandom tensor check")
        # logger.print("State:")
        # logger.print(one_sample.)

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

        latest_losses = losses[len(losses) - 1]
        if (config.max_steps > 0 and step >= config.max_steps) or (latest_losses < config.learning_rate):
            model.save_checkpoint(step)
            simulate_model(logger, config, game, model, step)
            break


def alpha_zero(config: Config):
    """Start all the worker processes for a full alphazero setup."""
    # game = pyspiel.load_game(config.game)
    game = tictactoe.TicTacToeGame()
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

    evaluators = []
    # evaluators = [spawn.Process(evaluator, kwargs={"game": game, "config": config,
    #                                                "num": i})
    #               for i in range(config.evaluators)]

    def broadcast(msg):
        for proc in actors + evaluators:
            proc.queue.put(msg)

    try:
      learner(game=game, config=config, actors=actors, evaluators=evaluators, broadcast_fn=broadcast)
    except (KeyboardInterrupt, EOFError):
      print("Caught a KeyboardInterrupt, stopping early.")
    finally:
      broadcast("")
      for proc in actors + evaluators:
        proc.join()

def simulate_model(logger, config, game, model, step=-1):
    with file_logger.FileLogger(config.path + '/log', 'preview_' + str(step), config.quiet) as plogger:
        # az_evaluator = evaluator_lib.AlphaZeroEvaluator(game, model)
        # random_evaluator = mcts.RandomRolloutEvaluator()
        # bots = [_init_bot(config, game, az_evaluator, False)]
        
        # for _ in range(1, game.num_players()):
        #   bots.append(mcts.MCTSBot(
        #         game,
        #         config.uct_c,
        #         config.max_simulations,
        #         random_evaluator,
        #         solve=True,
        #         verbose=False))
        az_evaluator = evaluator_lib.AlphaZeroEvaluator(game, model)
        evaluators = [az_evaluator.evaluate]
        prior_fns = [az_evaluator.prior]
        for _ in range(1, game.num_players()):
            evaluators.append(mcts_evaluation)
            prior_fns.append(mcts_prior)

        _play_game(plogger, 0, game, evaluators, prior_fns, 1, 0, True)

@watcher
def simulate_once(config, logger):
    # game = pyspiel.load_game(config.game)
    # game = tictactoe.TicTacToeGame()
    game = nimmt.NimmtGame()

    config = config._replace(
        observation_shape=game.observation_tensor_shape(),
        output_size=game.num_distinct_actions())
    model = _init_model_from_config(config)
    if config.cp_num and config.path:
        model.load_checkpoint(config.path + '/cp/checkpoint-' + str(config.cp_num))
    simulate_model(logger, config, game, model)

game_name='nimmt'
az_config = Config(
    game=game_name,
    cp_num=None,
    path='./sunday/' + game_name,
    learning_rate=0.001,
    weight_decay=1e-4,
    train_batch_size=2**10,
    replay_buffer_size=2**16,
    replay_buffer_reuse=3,
    max_steps=0,
    checkpoint_freq=10,

    actors=2,
    evaluators=1,
    uct_c=2,
    max_simulations=300,
    policy_alpha=1,
    policy_epsilon=0.25,
    temperature=1,
    temperature_drop=10,
    evaluation_window=50,
    eval_levels=7,

    nn_model="mlp",
    nn_width=128,
    nn_depth=30,
    observation_shape=None,
    output_size=None,

    quiet=True,
)

def main(unused_argv):
    # alpha_zero(config=az_config)
    simulate_once(config=az_config)

# simulate(config=az_config)
if __name__ == "__main__":
  with spawn.main_handler():
    app.run(main)
