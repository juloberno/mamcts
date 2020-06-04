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
#include "test/cost_constrained/cost_constrained_statistic_test_state.h"
#include <cstdio>

using namespace std;
using namespace mcts;



TEST(cost_constrained_statistic, action_risk_0) {
  int n_steps = 1;
  const Cost risk_action1 = 0.8f;
  const Reward goal_reward1 = 1.0f;
  const Cost risk_action2 = 0.3f;
  const Reward goal_reward2 = 0.1f;

  int num_samples = 1000;
  int num_collisions = 0;

  for(int i = 0; i < num_samples; ++i) {
    auto state = std::make_shared<CostConstrainedStatisticTestState>(n_steps, risk_action1, risk_action2,
                                            goal_reward1, goal_reward2, false);
    std::vector<Reward> rewards;
    Cost ego_cost = 0.0f;
    for( int y = 10; y < 100; ++y) {
      state = state->execute(JointAction{0}, rewards, ego_cost);
    }
    if(ego_cost > 0.0f) {
      num_collisions++;
    }
  }
  const double collision_risk = double(num_collisions)/num_samples;
  EXPECT_EQ(collision_risk, 0.0f); // ActionIdx=0 -> No collision should occur;
}

TEST(cost_constrained_statistic, action_risk_1) {
  int n_steps = 1;
  const Cost risk_action1 = 0.8f;
  const Reward goal_reward1 = 1.0f;
  const Cost risk_action2 = 0.3f;
  const Reward goal_reward2 = 0.1f;

  int num_samples = 1000;
  int num_collisions = 0;

  for(int i = 0; i < num_samples; ++i) {
    auto state = std::make_shared<CostConstrainedStatisticTestState>(n_steps, risk_action1, risk_action2,
                                            goal_reward1, goal_reward2, false);
    std::vector<Reward> rewards;
    Cost ego_cost = 0.0f;
    for( int y = 10; y < 100; ++y) {
      state = state->execute(JointAction{1}, rewards, ego_cost);
    }
    if(ego_cost > 0.0f) {
      num_collisions++;
    }
  }
  const double collision_risk = double(num_collisions)/num_samples;
  EXPECT_NEAR(collision_risk, risk_action1, 0.001); // ActionIdx=0 -> No collision should occur;
}
  


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();

}

