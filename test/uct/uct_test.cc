// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================

#include "gtest/gtest.h"

#define UNIT_TESTING
#define DEBUG
#define PLAN_DEBUG_INFO
#include "test/uct/uct_test_class.h"
#include "mcts/heuristics/random_heuristic.h"
#include "mcts/statistics/uct_statistic.h"
#include "test/uct/simple_state.h"
#include <cstdio>

using namespace std;
using namespace mcts;


MctsParameters default_uct_params() {
  MctsParameters parameters;
  parameters.DISCOUNT_FACTOR = 0.9;
  parameters.RANDOM_SEED = 1000;
  parameters.MAX_NUMBER_OF_ITERATIONS = 10000;
  parameters.MAX_SEARCH_TIME = 1000;
  parameters.MAX_SEARCH_DEPTH = 1000;
  
  parameters.random_heuristic.MAX_SEARCH_TIME = 10;
  parameters.random_heuristic.MAX_NUMBER_OF_ITERATIONS = 1000;

  parameters.uct_statistic.LOWER_BOUND = -1000;
  parameters.uct_statistic.UPPER_BOUND = 100;
  parameters.uct_statistic.EXPLORATION_CONSTANT = 0.7;

  return parameters;
}

TEST(test_mcts, verify_uct )
{
    Mcts<SimpleState, UctStatistic, UctStatistic, RandomHeuristic> mcts(default_uct_params());
    SimpleState state(4);
    
    mcts.search(state);

    UctTest test;
    test.verify_uct(mcts, 1000);

}

TEST(test_mcts, small_search_depth )
{
    auto params = default_uct_params();
    params.MAX_SEARCH_DEPTH = 2;
    Mcts<SimpleState, UctStatistic, UctStatistic, RandomHeuristic> mcts(params);
    SimpleState state(4);
    
    mcts.search(state);

    UctTest test;
}

TEST(test_mcts, generate_dot_file )
{
    Mcts<SimpleState, UctStatistic, UctStatistic, RandomHeuristic> mcts(default_uct_params());
    SimpleState state(4);
    
    mcts.search(state);
    mcts.printTreeToDotFile("test_tree");
}

TEST(test_mcts, vist_edges )
{
    Mcts<SimpleState, UctStatistic, UctStatistic, RandomHeuristic> mcts(default_uct_params());
    SimpleState state(4);
    
    mcts.search(state);
    const std::function<int(const SimpleState&, 
                        const SimpleState&,
                        const AgentIdx&)> edge_info_extractor = [](const SimpleState& start_state, 
                                        const SimpleState& end_state,
                                        const AgentIdx& agent_idx) {
      return start_state.get_state_length() - end_state.get_state_length();
    };
    const auto edge_info = mcts.visit_mcts_tree_edges(edge_info_extractor);
}



int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();

}