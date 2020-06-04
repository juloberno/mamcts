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
  CostConstrainedStatistic stat_parent(5, 1, mcts_default_parameters());
}

TEST(cost_constrained_mcts, one_step_higher_reward_higher_risk_constraint_eq) {
  int n_steps = 1;
  const Cost risk_action1 = 0.8;
  const Reward goal_reward1 = 1000;
  const Cost risk_action2 = 0.3;
  const Reward goal_reward2 = 10;

  CostConstrainedStatisticTestState state(n_steps, risk_action1, risk_action2,
                                         goal_reward1, goal_reward2, false);
  auto mcts_parameters = mcts_default_parameters();
  // collision risk 1 is higher but within constraint
  mcts_parameters.cost_constrained_statistic.COST_CONSTRAINT = risk_action1;
   Mcts<CostConstrainedStatisticTestState, CostConstrainedStatistic,
               UctStatistic, RandomHeuristic> mcts(mcts_parameters);
  mcts.search(state);
  auto best_action = mcts.returnBestAction();
  EXPECT_EQ(best_action, 1);
}

TEST(cost_constrained_mcts, one_step_higher_reward_higher_risk_constraint_low) {
    int n_steps = 1;
  const Cost risk_action1 = 0.8;
  const Reward goal_reward1 = 1000;
  const Cost risk_action2 = 0.3;
  const Reward goal_reward2 = 10;

  CostConstrainedStatisticTestState state(n_steps, risk_action1, risk_action2,
                                         goal_reward1, goal_reward2, false);
  auto mcts_parameters = mcts_default_parameters();
  // collision risk 1 is higher but within constraint
  mcts_parameters.cost_constrained_statistic.COST_CONSTRAINT = risk_action1 - 0.05;
  mcts_parameters.MAX_SEARCH_DEPTH
   Mcts<CostConstrainedStatisticTestState, CostConstrainedStatistic,
               UctStatistic, RandomHeuristic> mcts(mcts_parameters);
  mcts.search(state);
  auto best_action = mcts.returnBestAction();
  EXPECT_EQ(best_action, 2);
}


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();

}

