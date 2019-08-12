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





TEST(test_execute, verify_state){
    SimpleState state(2);
    //Setup with similar actions
    JointAction ja=std::vector<ActionIdx> {1,1};
    JointAction *jaPtr = &ja;
    std::vector<Reward> rewards = {0,0};
    std::vector<Reward> *rPtr = &rewards;
    std::string before = state.sprintf();
    state=*state.execute(*jaPtr,*rPtr);
    std::string after = state.sprintf();
    EXPECT_EQ(before,after);
    EXPECT_EQ(rewards[0],0);
    //different actions
    ja=std::vector<ActionIdx> {1,0};
    state=*state.execute(*jaPtr, *rPtr);
    after = state.sprintf();
    EXPECT_NE(before,after);
    EXPECT_EQ(rewards[0],1);    
    //testing final reward
    SimpleState state_2(9);
    state_2=*state_2.execute(*jaPtr, *rPtr);
    EXPECT_EQ(rewards[0],10);
}


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();

}