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
#include "test/cost_constrained/cost_constrained_statistic_test_state_multiple_cost.h"
#include "mcts/heuristics/random_heuristic.h"
#include "mcts/statistics/random_actions_statistic.h"
#include "mcts/mcts.h"
#include <cstdio>

using namespace std;
using namespace mcts;


struct CostConstrainedTest : public ::testing::Test {
  CostConstrainedTest() {}
  virtual ~CostConstrainedTest() {}

  virtual void TearDown() {
        delete mcts_;
        delete state_;
    }

  void SetUp( int n_steps, Reward goal_reward1, Reward goal_reward2,
             Cost risk_action1, Cost risk_action2, std::vector<Cost> cost_constraints,
             double lambda_init, bool widening, unsigned int number_of_iters=3000) {
            n_steps_ = n_steps;
            goal_reward1_ = goal_reward1;
            goal_reward2_ = goal_reward2;
            risk_action1_ = risk_action1;
            risk_action2_ = risk_action2;
            lambda_init_ = lambda_init;
            state_ = new CostConstrainedStatisticTestStateMultipleCost(n_steps, risk_action1, risk_action2,
                                                  goal_reward1, goal_reward2, false);

            mcts_parameters_ = mcts_default_parameters();
            mcts_parameters_.cost_constrained_statistic.COST_CONSTRAINTS = cost_constraints;
            mcts_parameters_.cost_constrained_statistic.REWARD_UPPER_BOUND = std::max(goal_reward1_, goal_reward2_);
            mcts_parameters_.cost_constrained_statistic.REWARD_LOWER_BOUND = 0.0f;
            mcts_parameters_.cost_constrained_statistic.COST_LOWER_BOUND = 0.0f;
            mcts_parameters_.cost_constrained_statistic.COST_UPPER_BOUND = 1.0f;
            mcts_parameters_.cost_constrained_statistic.KAPPA = 10.0f;
            mcts_parameters_.cost_constrained_statistic.GRADIENT_UPDATE_STEP = 1.5f;
            mcts_parameters_.cost_constrained_statistic.TAU_GRADIENT_CLIP = 1.0f;
            mcts_parameters_.cost_constrained_statistic.ACTION_FILTER_FACTOR = 0.5f;
            mcts_parameters_.cost_constrained_statistic.USE_LAMBDA_POLICY = true;
            mcts_parameters_.cost_constrained_statistic.USE_CHANCE_CONSTRAINED_UPDATES = {false, true};
            mcts_parameters_.DISCOUNT_FACTOR = 0.9;
            mcts_parameters_.MAX_SEARCH_TIME = 1000000000;
            mcts_parameters_.MAX_NUMBER_OF_ITERATIONS = number_of_iters;
            mcts_parameters_.NUM_PARALLEL_MCTS = 4;
            if (widening) {
              mcts_parameters_.random_actions_statistic.PROGRESSIVE_WIDENING_ALPHA = 0.25;
              mcts_parameters_.random_actions_statistic.PROGRESSIVE_WIDENING_K = 2.0;
            } else {
              mcts_parameters_.random_actions_statistic.PROGRESSIVE_WIDENING_ALPHA = 1.0;
              mcts_parameters_.random_actions_statistic.PROGRESSIVE_WIDENING_K = 1.0;
            }

            mcts_parameters_.cost_constrained_statistic.LAMBDAS = {lambda_init, lambda_init};
            mcts_ = new Mcts<CostConstrainedStatisticTestStateMultipleCost, CostConstrainedStatistic,
                        RandomActionsStatistic, RandomHeuristic>(mcts_parameters_);
    }

    CostConstrainedStatisticTestStateMultipleCost* state_;
    Mcts<CostConstrainedStatisticTestStateMultipleCost, CostConstrainedStatistic,
                        RandomActionsStatistic, RandomHeuristic>* mcts_;
    MctsParameters mcts_parameters_;
    int n_steps_;
    Reward goal_reward1_;
    Reward goal_reward2_;
    Cost risk_action1_;
    Cost risk_action2_;
    double lambda_init_;

};


struct CostConstrainedNStepTest : public CostConstrainedTest {
  CostConstrainedNStepTest() {}
  virtual ~CostConstrainedNStepTest() {}

  auto make_initial_state(unsigned int seed) const {
     return std::make_shared<CostConstrainedStatisticTestStateMultipleCost>(n_steps_, risk_action1_, risk_action2_,
                                            goal_reward1_, goal_reward2_, false, seed);
  }
};


TEST_F(CostConstrainedTest, allow_collision_and_safety_violation) {
  SetUp(1, 2.0f, 0.5f, 0.8f, 0.3f, {0.4f, 0.8f}, 2.2f, false, 30);

  mcts_->search(*state_);
  auto best_action = mcts_->returnBestAction();
  const auto root = mcts_->get_root();
  const auto& reward_stats = root.get_ego_int_node().get_reward_ucb_statistics();
  const auto& cost_stats1 = root.get_ego_int_node().get_cost_ucb_statistics(0);
  const auto& cost_stats2 = root.get_ego_int_node().get_cost_ucb_statistics(1);

  // Cost statistics desired
  EXPECT_NEAR(cost_stats1.at(0).action_value_, 0.0, 0.0);
  EXPECT_NEAR(cost_stats1.at(1).action_value_, 0.0, 0.05);
  EXPECT_NEAR(cost_stats1.at(2).action_value_, 0.3, 0.06);

  EXPECT_NEAR(cost_stats2.at(0).action_value_, 0.0, 0.0);
  EXPECT_NEAR(cost_stats2.at(1).action_value_, 0.8, 0.05);
  EXPECT_NEAR(cost_stats2.at(2).action_value_, 0.0, 0.00);

  // Reward statistics desired
  EXPECT_NEAR(reward_stats.at(2).action_value_, (1-risk_action2_)*goal_reward2_, 0.09);
  EXPECT_NEAR(reward_stats.at(1).action_value_, (1-risk_action1_)*goal_reward1_, 0.08);
  EXPECT_NEAR(reward_stats.at(0).action_value_, 0, 0.00);

  EXPECT_EQ(best_action, 2);

  LOG(INFO) << "\n"  << root.get_ego_int_node().print_edge_information(0);
}


int main(int argc, char **argv) {
  FLAGS_alsologtostderr = true;
  FLAGS_v = 0;
  google::InitGoogleLogging("test");
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();

}

