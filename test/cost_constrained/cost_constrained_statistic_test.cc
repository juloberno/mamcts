// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================

#include "gtest/gtest.h"
#include <gtest/gtest_prod.h>

#define UNIT_TESTING
#define DEBUG
#define PLAN_DEBUG_INFO
#include "mcts/cost_constrained/cost_constrained_statistic.h"
#include "test/cost_constrained/cost_constrained_statistic_test_state.h"
#include "mcts/heuristics/random_heuristic.h"
#include <cstdio>

using namespace std;
using namespace mcts;



TEST(cost_constrained_statistic, backprop_cost_reward_updates) {

  CostConstrainedStatisticTestState state(5);
  CostConstrainedStatistic stat_parent(5, 1, mcts_default_parameters());

}

TEST(cost_constrained_statistic, search) {

  CostConstrainedStatisticTestState state(5);
   Mcts<CostConstrainedStatisticTestState, CostConstrainedStatistic,
               UctStatistic, RandomHeuristic> mcts(mcts_default_parameters());

  mcts.search(state);
}


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();

}

