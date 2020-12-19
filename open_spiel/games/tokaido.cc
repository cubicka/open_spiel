// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/games/tokaido.h"

#include <sys/types.h>

#include <algorithm>
#include <utility>

#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/utils/tensor_view.h"

namespace open_spiel {
namespace tokaido {

namespace {
// Facts about the game
const GameType kGameType{
    /*short_name=*/"tokaido",
    /*long_name=*/"Tokaido",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/10,
    /*min_num_players=*/10,
    /*provides_information_state_string=*/false,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/ {}
};

static std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new TokaidoGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);
}  // namespace

int CardScore(int card) {
  if (card == 55) return 7;
  else if (card%11 == 0) return 5;
  else if (card%10 == 0) return 3;
  else if (card%5 == 0) return 2;
  else return 1;
}

bool compareCardStacks(std::array<int, 5> &a, std::array<int, 5> &b) {
  return a[0] < b[0];
}

TokaidoState::TokaidoState(std::shared_ptr<const Game> game): State(game) {
  for (int i=0; i < 104; i++) shuffledCards[i] = i + 1;
  std::random_device rng;
  std::mt19937 urng(rng());
  std::shuffle(shuffledCards, shuffledCards + 104, urng);
  // std::random_device rng = std::uniform_real_distribution<double>(0., 1.);
  numPlayers = (rng() % 9) + 2;

  for (int i=0; i < numPlayers; i++) {
    for (int k=0; k < 10; k++) {
      playerHands[i][k] = shuffledCards[i*10 + k];
    }
    std::sort(begin(playerHands[i]), end(playerHands[i]));

    for (int k=0; k<=104; k++) bulls[i][k] = false;
  }

  for (int i=0; i<4; i++) {
    for (int k=0; k<5; k++) cardStacks[i][k] = -1;
    cardStacks[i][0] = shuffledCards[numPlayers * 10 + i];
  }
  std::sort(begin(cardStacks), end(cardStacks), compareCardStacks);

  currMode = 0;
  currPlayer = 0;
}

bool TokaidoState::IsTerminal() const {
  return currMode == -1;
}

int TokaidoState::CurrentPlayer() const {
  return IsTerminal() ? kTerminalPlayerId : currPlayer;
}

std::vector<double> TokaidoState::Returns() const {
  if (!IsTerminal()) {
    return std::vector<double>(10, 0.0);
  }

  std::vector<double> returns(10, -1.0);
  int bestScores = scores[0];
  for (int i = 0; i < numPlayers; i++) bestScores = std::min(bestScores, scores[i]);
  for (int i = 0; i < numPlayers; i++) if (scores[i] <= bestScores) returns[i] = 1.0;
  return returns;
}

std::vector<Action> TokaidoState::LegalActions() const {
  std::vector<Action> actions;
  if (currMode == 0) {
    for (int i = 0; i < 10; i++) {
      if (playerHands[currPlayer][i] != -1) actions.push_back(playerHands[currPlayer][i]);
    }
  } else {
    actions.reserve(4);
    for (int i = 0; i < 4; i++) actions.emplace_back(105 + i);
  }

  return actions;
}

std::string TokaidoState::ActionToString(Player player, Action move_id) const {
  if (move_id > 104) return absl::StrCat("Take stack #", move_id);
  return absl::StrCat("Play card #", move_id);
}

std::string BullsToString(const TokaidoState  &state) {
  std::string rv = "Bulls:\n";
  for (int i=0; i<state.numPlayers; i++) {
    absl::StrAppend(&rv, "  Bulls #", i, ":");
    int total = 0;
    for (int k=1; k<=104; k++) if (state.bulls[i][k]) {
      absl::StrAppend(&rv, " ", k);
      total += CardScore(k);
    }

    absl::StrAppend(&rv, "\n  Total #", i, ":", total, "\n");
  } 

  return rv;
}

std::string BoardToString(const TokaidoState &state) {
  std::string rv = "Board:\n";
  for(int i=0; i<4; i++) {
    absl::StrAppend(&rv, "  ");
    for (int k=0; k<5; k++) {
      if (state.cardStacks[i][k] != -1) absl::StrAppend(&rv, state.cardStacks[i][k], " ");
    }
    absl::StrAppend(&rv, "\n");
  }

  return rv;
}

std::string HandsToString(const TokaidoState &state) {
  std::string rv = "Hands:\n";
  for (int i=0; i<state.numPlayers; i++) {
    absl::StrAppend(&rv, "  ", i, ":");
    for (int k=0; k<10; k++) {
      if (state.playerHands[i][k] != -1) absl::StrAppend(&rv, " ", state.playerHands[i][k]);
    }
    absl::StrAppend(&rv, "\n");
  }

  return rv;
}

std::string StackToString(const TokaidoState &state) {
  std::string rv = "Stacks:";
  int start = (state.currMode == 0 ? 0 : state.currStack); 
  int end = (state.currMode==0? state.currPlayer: state.numPlayers);
  for (int i=start; i<end; i++) {
    absl::StrAppend(&rv, " (", state.cardsToBePlaced[i][0], ",", state.cardsToBePlaced[i][1], ")");
  }
  absl::StrAppend(&rv, "\n");
  return rv;
}

std::string TokaidoState::ToString() const {
  if (IsTerminal()) {
    std::vector<double> returns = Returns();
    std::string rv = "Winners are:";
    for(int i=0; i<numPlayers; i++) if (returns[i] > 0) absl::StrAppend(&rv, i, " ");
    absl::StrAppend(&rv, "\n");
    absl::StrAppend(&rv, BullsToString(*this));
    return rv;
  } else if (currMode == 0 || currMode == 1) {
    return absl::StrCat(BoardToString(*this), StackToString(*this), HandsToString(*this), BullsToString(*this));
  } else return "Invalid State";
}

std::string TokaidoState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void TokaidoState::ObservationTensor(Player player,
                                 absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  // 0..9 - Bulls
  // 10 - Hands
  // 11 - Stack to be Placed
  // 12..15 - Current Board
  // 16 - Owner of the Stack to be Placed (10 * 10)
  // 17 - Player Count (10)
  TensorView<2> view(values, {18, 104}, true);
  std::fill(values.begin(), values.end(), 0.);

  for(int i=0; i<numPlayers; i++) {
    for(int k=1; k<=104; k++) if (bulls[i][k]) view[{i, k-1}] = 1.0;
  }

  for(int i=0; i<10; i++) if (playerHands[player][i] != -1) {
    view[{10, playerHands[player][i]-1}] = 1.0;
  }

  if (currMode == 1) {
    for (int i=currStack; i<numPlayers; i++) {
      view[{11, cardsToBePlaced[i][0]-1}] = 1.0;
      view[{16, (i-currStack)*10 + cardsToBePlaced[i][1]}] = 1.0;
    }
  }

  for (int i=0; i<4; i++) for (int k=0; k<5; k++) if (cardStacks[i][k] != -1) {
    view[{12+i, cardStacks[i][k]-1}] = 1.0;
  }

  for (int i=0; i<numPlayers; i++) view[{17, i}] = -1.0;
  view[{17, player}] = 1.0;
}

std::unique_ptr<State> TokaidoState::Clone() const {
  return std::unique_ptr<State>(new TokaidoState(*this));
}

void SwapBull(TokaidoState &state, int row) {
  int player = state.cardsToBePlaced[state.currStack][1];
  int card = state.cardsToBePlaced[state.currStack][0];
  for (int i=0; i<5; i++) if(state.cardStacks[row][i] != -1) {
    state.bulls[player][state.cardStacks[row][i]] = true;
  }

  for (int i=0; i<5; i++) state.cardStacks[row][i] = -1;
  state.cardStacks[row][0] = card;

  std::sort(begin(state.cardStacks), end(state.cardStacks), compareCardStacks);
  state.currStack++;
}

std::pair<int,int> PlacementIndex(TokaidoState &state, int card) {
  int placementIdx = -1, placementCol = -1;
  for (int i=0; i<4; i++) {
    int currCol = 0;
    for (int k=0; k<5; k++) if (state.cardStacks[i][k] != -1) currCol = k;
    if (state.cardStacks[i][currCol] < card) {
      if (placementIdx == -1 || state.cardStacks[placementIdx][placementCol] < state.cardStacks[i][currCol]) {
        placementIdx = i; placementCol = currCol;
      }
    }
  }

  return std::make_pair(placementIdx, placementCol);
}

void PlaceOnStack(TokaidoState &state) {
  while(state.currStack < state.numPlayers) {
    state.currPlayer = state.cardsToBePlaced[state.currStack][1];
    std::pair<int, int> idx = PlacementIndex(state, state.cardsToBePlaced[state.currStack][0]);
    if (idx.first == -1) {
      return;
    }

    if (idx.second >= 4) {
      SwapBull(state, idx.first);
    } else {
      state.cardStacks[idx.first][idx.second+1] = state.cardsToBePlaced[state.currStack][0];
      state.currStack++;
    }
  }

  bool canStillPlay = false;
  for (int i=0; i<10; i++) if(state.playerHands[0][i] != -1) canStillPlay = true;

  if (canStillPlay) {
    state.currMode = 0;
    state.currPlayer = 0;
  } else {
    state.currMode = -1;
    for(int i=0; i<state.numPlayers; i++) {
      state.scores[i] = 0;
      for (int k=1; k<=104; k++) if(state.bulls[i][k]) {
        state.scores[i] += CardScore(k);
      }
    }
  }
}

void TokaidoState::DoApplyAction(Action move) {
  if (currMode == 0 && move >= 1 && move <= 104) {
    cardsToBePlaced[currPlayer][0] = move;
    cardsToBePlaced[currPlayer][1] = currPlayer;
    for(int i=0; i<10; i++) if(playerHands[currPlayer][i] == move) playerHands[currPlayer][i] = -1;
    currPlayer++;

    if (currPlayer >= numPlayers) {
      currMode = 1;
      currStack = 0;
      std::sort(begin(cardsToBePlaced), begin(cardsToBePlaced) + numPlayers);
      PlaceOnStack(*this);
    }
  } else if (currMode == 1 && move >= 105 && move <= 108) {
    SwapBull(*this, move-105);
    PlaceOnStack(*this);
  } else {
    SpielFatalError(absl::StrCat("Move ", move, " is invalid."));
  }
}

TokaidoGame::TokaidoGame(const GameParameters& params): Game(kGameType, params) {}
}  // namespace pig
}  // namespace open_spiel
