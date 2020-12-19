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

#ifndef OPEN_SPIEL_GAMES_TOKAIDO_H_
#define OPEN_SPIEL_GAMES_TOKAIDO_H_

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

// A simple jeopardy dice game that includes chance nodes.
// See http://cs.gettysburg.edu/projects/pig/index.html for details.
// Also https://en.wikipedia.org/wiki/Pig_(dice_game)
//
// Parameters:
//     "diceoutcomes"  int    number of outcomes of the dice  (default = 6)
//     "horizon"       int    max number of moves before draw (default = 1000)
//     "players"       int    number of players               (default = 2)
//     "winscore"      int    number of points needed to win   (default = 100)

namespace open_spiel {
namespace tokaido {

class TokaidoGame;

class TokaidoState : public State {
 public:
  TokaidoState(const TokaidoState&) = default;
  TokaidoState(std::shared_ptr<const Game> game);

  Player CurrentPlayer() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;

  std::string ToString() const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;

  std::unique_ptr<State> Clone() const override;

  std::vector<Action> LegalActions() const override;
  std::string ActionToString(Player player, Action move_id) const override;

  // std::string BullsToString() const; 
  // std::string BoardToString() const;
  // std::string HandsToString() const;
  // std::string StackToString() const;
  // std::pair<int,int> PlacementIndex(int card) const;
  // void PlaceOnStack();
  // void SwapBull(int row);
  // int CardScore(int card) const;

 int numPlayers;
 int shuffledCards[104];
 std::array<std::array<int, 10>, 10> playerHands;
 std::array<std::array<int, 2>, 10> cardsToBePlaced;
 std::array<std::array<int, 5>, 4> cardStacks;
 int scores[10];
 bool bulls[10][105];

  // 0 - Pick card to be placed at the table
  // 1 - Pick a 5-card stack to be taken from table
  int currMode;
  int currPlayer;
  int currStack;

 protected:
  void DoApplyAction(Action move_id) override;

};

class TokaidoGame : public Game {
 public:
  explicit TokaidoGame(const GameParameters& params);
  int NumDistinctActions() const override { return 109; }
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new TokaidoState(shared_from_this()));
  }


  // There is arbitrarily chosen number to ensure the game is finite.
  int MaxGameLength() const override { return 300; }
  int NumPlayers() const override { return 10; }
  double MinUtility() const override { return -1; }
  double UtilitySum() const override { return 0; }
  double MaxUtility() const override { return +1; }
  std::vector<int> ObservationTensorShape() const override {
    return {18, 104, 1};
  }
};

}  // namespace tokaido
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_TOKAIDO_H_
