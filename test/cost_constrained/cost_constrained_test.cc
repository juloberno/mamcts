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
#include "mcts/statistics/random_actions.h"
#include <cstdio>

using namespace std;
using namespace mcts;



TEST(cost_constrained_statistic, backprop_cost_reward_updates) {
  CostConstrainedStatistic stat_parent(5, 1, mcts_default_parameters());
}

TEST(cost_constrained_mcts, one_step_higher_reward_higher_risk_constraint_eq) {
  FLAGS_alsologtostderr = true;
  FLAGS_v = 5;
  google::InitGoogleLogging("test");
  int n_steps = 1;
  const Cost risk_action1 = 0.8f;
  const Reward goal_reward1 = 2.0f; // gives expectation 2*0.2 = 0.4 <- only slightly better
  const Cost risk_action2 = 0.3f;
  const Reward goal_reward2 = 0.5f; // gives expectation 0.5*0.7 = 0.35

  CostConstrainedStatisticTestState state(n_steps, risk_action1, risk_action2,
                                         goal_reward1, goal_reward2, false);
  auto mcts_parameters = mcts_default_parameters();
  // collision risk 1 is higher but within constraint
  mcts_parameters.cost_constrained_statistic.COST_CONSTRAINT = risk_action1;
  mcts_parameters.cost_constrained_statistic.REWARD_UPPER_BOUND = 10.0f;
  mcts_parameters.cost_constrained_statistic.REWARD_LOWER_BOUND = 0.0f;
  mcts_parameters.cost_constrained_statistic.COST_LOWER_BOUND = 0.0f;
  mcts_parameters.cost_constrained_statistic.COST_UPPER_BOUND = 1.0f;
  mcts_parameters.cost_constrained_statistic.EXPLORATION_CONSTANT = 0.7f;
  mcts_parameters.cost_constrained_statistic.GRADIENT_UPDATE_STEP = 0.1f;
  mcts_parameters.cost_constrained_statistic.TAU_GRADIENT_CLIP = 1.0f;
  mcts_parameters.cost_constrained_statistic.ACTION_FILTER_FACTOR = 1.0f;
  mcts_parameters.DISCOUNT_FACTOR = 0.9;
  mcts_parameters.MAX_SEARCH_TIME = 1000000000;
  mcts_parameters.MAX_NUMBER_OF_ITERATIONS = 1000;

    // Lambda desired
  const double lambda_desired_max = ( (1 - risk_action1) * goal_reward1 - ( 1 - risk_action2) * goal_reward2 ) /
                            (risk_action2 - risk_action1);

  mcts_parameters.cost_constrained_statistic.LAMBDA = lambda_desired_max;
  Mcts<CostConstrainedStatisticTestState, CostConstrainedStatistic,
               RandomActions, RandomHeuristic> mcts(mcts_parameters);
  mcts.search(state);
  auto best_action = mcts.returnBestAction();
  const auto root = mcts.get_root();
  const auto& reward_stats = root.get_ego_int_node().get_reward_ucb_statistics();
  const auto& cost_stats = root.get_ego_int_node().get_cost_ucb_statistics();


  EXPECT_TRUE(mcts_parameters.cost_constrained_statistic.LAMBDA <= lambda_desired_max);

  // Cost statistics desired
  EXPECT_NEAR(cost_stats.at(2).action_value_, risk_action2, 0.05);
  EXPECT_NEAR(cost_stats.at(1).action_value_, risk_action1, 0.05);
  EXPECT_NEAR(cost_stats.at(0).action_value_, 0, 0.00);

  // Reward statistics desired
  EXPECT_NEAR(reward_stats.at(2).action_value_, (1-risk_action2)*goal_reward2, 0.05);
  EXPECT_NEAR(reward_stats.at(1).action_value_, (1-risk_action1)*goal_reward1, 0.05);
  EXPECT_NEAR(reward_stats.at(0).action_value_, 0, 0.00);

  EXPECT_EQ(best_action, 1);
}



int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();

}

