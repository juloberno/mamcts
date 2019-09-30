// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================

#include "gtest/gtest.h"

#define UNIT_TESTING
#define DEBUG
#define PLAN_DEBUG_INFO
#include "mcts/heuristics/random_heuristic.h"
#include "mcts/hypothesis/hypothesis_statistic.h"
#include "mcts/statistics/uct_statistic.h"
#include "test/hypothesis/hypothesis_simple_state.h"
#include <cstdio>

using namespace std;
using namespace mcts;


std::mt19937  mcts::RandomGenerator::random_generator_;

TEST(test_mcts, verify_uct )
{

    RandomGenerator::random_generator_ = std::mt19937(1000);
    Mcts<HypothesisSimpleState, UctStatistic, HypothesisStatistic, RandomHeuristic> mcts;
    HypothesisSimpleState state(4);
    
    mcts.search(state, 50000, 2000);

}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();

}