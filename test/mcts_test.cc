// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================

#include "gtest/gtest.h"

#define UNIT_TESTING
#define DEBUG
#define PLAN_DEBUG_INFO
#include "test/mcts_test_class.h"
#include "mcts/heuristics/random_heuristic.h"
#include "mcts/statistics/uct_statistic.h"
#include "test/simple_state.h"
#include <cstdio>

using namespace std;
using namespace mcts;


std::mt19937  mcts::RandomGenerator::random_generator_;

TEST(test_mcts, verify_uct )
{

    RandomGenerator::random_generator_ = std::mt19937(1000);

    Mcts<SimpleState, UctStatistic, UctStatistic, RandomHeuristic> mcts;

    SimpleState state(4);
    
    mcts.search(state);
    mcts.generateTree("test_tree");
    std::cout << "Finished search." << std::endl;

    MctsTest test;
   test.verify_uct(mcts,1);
    std::cout << "Finished verification." << std::endl;

}



int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();

}