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
"""An MCTS Evaluator for an AlphaZero model."""

import numpy as np

from eval import Evaluator
import utils.lru_cache as lru_cache


class AlphaZeroEvaluator(Evaluator):
  """An AlphaZero MCTS Evaluator."""

  def __init__(self, game, model, cache_size=2**16):
    """An AlphaZero MCTS Evaluator."""
    # if game.num_players() != 2:
    #   raise ValueError("Game must be for two players.")
    # game_type = game.get_type()
    # if game_type.reward_model != pyspiel.GameType.RewardModel.TERMINAL:
    #   raise ValueError("Game must have terminal rewards.")
    # if game_type.dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
    #   raise ValueError("Game must have sequential turns.")
    # if game_type.chance_mode != pyspiel.GameType.ChanceMode.DETERMINISTIC:
    #   raise ValueError("Game must be deterministic.")

    self._model = model
    self._cache = lru_cache.LRUCache(cache_size)
    self._policy_cache = lru_cache.LRUCache(cache_size)

  def cache_info(self):
    return self._cache.info()

  def clear_cache(self):
    self._cache.clear()
    self._policy_cache.clear()

  def _inference(self, state, player):
    # Make a singleton batch
    obs = np.expand_dims(state.observation_tensor(player), 0)
    mask = np.expand_dims(state.legal_actions_mask(player), 0)

    # ndarray isn't hashable
    cache_key = obs.tobytes() + mask.tobytes()

    value, policy = self._cache.make(
        cache_key, lambda: self._model.inference(obs, mask))

    return value[0, 0], policy[0]  # Unpack batch

  def evaluate(self, state, player):
    """Returns a value for the given state."""
    value, _ = self._inference(state, player)
    # return np.array([value, -value])
    return value

  def prior(self, state):
    """Returns the probabilities for all actions."""
    _, policy = self._inference(state, state.current_player())
    return [(action, policy[action]) for action in state.legal_actions()]

  def reuse_policy(self, player, state):
    # Make a singleton batch
    obs = np.expand_dims(state.observation_tensor(player), 0)
    mask = np.expand_dims(state.legal_actions_mask(player), 0)

    # ndarray isn't hashable
    cache_key = obs.tobytes() + mask.tobytes()
    return self._policy_cache.get(cache_key)

  def save_policy(self, player, state, policy, total_reward, visit_count, outcome):
    # Make a singleton batch
    obs = np.expand_dims(state.observation_tensor(player), 0)
    mask = np.expand_dims(state.legal_actions_mask(player), 0)

    # ndarray isn't hashable
    cache_key = obs.tobytes() + mask.tobytes()
    return self._policy_cache.set(cache_key, (policy, total_reward, visit_count, outcome))
