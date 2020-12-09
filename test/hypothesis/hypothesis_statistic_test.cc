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
#include "mcts/hypothesis/hypothesis_statistic.h"
#include "test/hypothesis/hypothesis_statistic_test_state.h"
#include "mcts/hypothesis/hypothesis_belief_tracker.h"
#include <cstdio>

using namespace std;
using namespace mcts;



TEST(hypothesis_statistic, backprop_hypothesis_action_selection) {
  std::unordered_map<AgentIdx, HypothesisId> current_agents_hypothesis = {
      {1,0}, {2,1}
  };

  // First iteration with hypothesis 0 for agent 1
  HypothesisStatisticTestState state(current_agents_hypothesis);
  HypothesisStatistic stat_parent(5, 1, mcts_default_parameters()); // agents 1 statistic 
  auto action_idx = stat_parent.choose_next_action(state);
  EXPECT_EQ(action_idx, 5);
  stat_parent.collect( 1, EgoCosts{2.3f, 0.0f}, action_idx, {0,0});

  HypothesisStatistic heuristic(5,1, mcts_default_parameters());
  heuristic.set_heuristic_estimate(10.0f , EgoCosts{20.0f, 1.2323f});

  HypothesisStatistic stat_child(5,1, mcts_default_parameters());
  stat_child.update_from_heuristic(heuristic);
  stat_parent.update_statistic(stat_child);

  const auto ucb_stats =stat_parent.get_ucb_statistics();
  const auto node_counts = stat_parent.get_total_node_visits();

  EXPECT_NEAR(ucb_stats.at(0).at(action_idx).action_ego_cost_, 2.3f
                    +mcts_default_parameters().DISCOUNT_FACTOR*(20.0f+1.2323f), 0.001);
  EXPECT_EQ(ucb_stats.at(0).at(action_idx).action_count_, 1);
  EXPECT_EQ(node_counts.at(0), 1);

  // Second update with hypothesis 0 for agent 1, action is the same as only one action available
  HypothesisStatistic heuristic2(5,1, mcts_default_parameters());
  heuristic2.set_heuristic_estimate(15.0f , EgoCosts{24.5f, 0.0f});

  HypothesisStatistic stat_child2(5,1, mcts_default_parameters());
  stat_child2.update_from_heuristic(heuristic2);
  auto action_idx2 = stat_parent.choose_next_action(state);
  EXPECT_EQ(action_idx2, 5);
  stat_parent.collect( 1, EgoCosts{4.3f, 0.0f}, action_idx2, {0,0});
  stat_parent.update_statistic(stat_child2);

  const auto ucb_stats2 =stat_parent.get_ucb_statistics();
  const auto node_counts2 = stat_parent.get_total_node_visits();

  EXPECT_NEAR(ucb_stats2.at(0).at(action_idx2).action_ego_cost_, (2.3f+4.3
                    +mcts_default_parameters().DISCOUNT_FACTOR*(20.0f+1.2323f)+
                    mcts_default_parameters().DISCOUNT_FACTOR*24.5f)/2, 0.001);
  EXPECT_EQ(ucb_stats2.at(0).at(action_idx2).action_count_, 2);
  EXPECT_EQ(node_counts2.at(0), 2);

  // Third update with changed actions hypothesis 0 for agent 1
  state.change_actions();
  HypothesisStatistic heuristic3(5,1, mcts_default_parameters());
  heuristic3.set_heuristic_estimate(15.0f , EgoCosts{450.5f, 0.0f});

  HypothesisStatistic stat_child3(5,1, mcts_default_parameters());
  stat_child3.update_from_heuristic(heuristic3);
  auto action_idx3 = stat_parent.choose_next_action(state);
  EXPECT_EQ(action_idx3, 3);
  stat_parent.collect( -1, EgoCosts{1000.3f, 0.0f}, action_idx3, {0,0});
  stat_parent.update_statistic(stat_child3);

  const auto ucb_stats3 =stat_parent.get_ucb_statistics();
  const auto node_counts3 = stat_parent.get_total_node_visits();

  EXPECT_NEAR(ucb_stats3.at(0).at(action_idx3).action_ego_cost_, 1000.3f +
                           mcts_default_parameters().DISCOUNT_FACTOR*450.5f, 0.001);
  EXPECT_EQ(ucb_stats3.at(0).at(action_idx3).action_count_, 1);
  EXPECT_EQ(node_counts3.at(0), 3);

  // Fourth update with hypothesis 1 for agent 1 
  current_agents_hypothesis = {
      {1,1}, {2,1}
  }; //< state holds a refence to current selected hypothesis

  HypothesisStatistic heuristic4(5,1, mcts_default_parameters());
  heuristic4.set_heuristic_estimate(15.0f , EgoCosts{45.5f, 0.0f});

  HypothesisStatistic stat_child4(5,1, mcts_default_parameters());
  stat_child4.update_from_heuristic(heuristic4);
  auto action_idx4 = stat_parent.choose_next_action(state);
  EXPECT_EQ(action_idx4, 4);
  stat_parent.collect( -1, EgoCosts{10.3f, 0.0f}, action_idx4, {0,0});
  stat_parent.update_statistic(stat_child4);

  const auto ucb_stats4 =stat_parent.get_ucb_statistics();
  const auto node_counts4 = stat_parent.get_total_node_visits();

  EXPECT_NEAR(ucb_stats4.at(1).at(action_idx4).action_ego_cost_, 10.3f +
                           mcts_default_parameters().DISCOUNT_FACTOR*45.5f, 0.001);
  EXPECT_EQ(ucb_stats4.at(1).at(action_idx4).action_count_, 1);
  EXPECT_EQ(node_counts4.at(1), 1);


}

TEST(hypothesis_statistic, backprop_heuristic_hyp1) {
  // Now test something for agent 2
  const std::unordered_map<AgentIdx, HypothesisId> current_agents_hypothesis = {
      {1,0}, {2,1}
  };
  HypothesisStatisticTestState state(current_agents_hypothesis);
  HypothesisStatistic stat_parent(5,2, mcts_default_parameters()); // agents 2 statistic 
  auto action_idx = stat_parent.choose_next_action(state);
  stat_parent.collect( 1, EgoCosts{5.3f, 0.0f}, action_idx, {0,0});

  HypothesisStatistic heuristic(5,2, mcts_default_parameters());
  heuristic.set_heuristic_estimate(10.0f , EgoCosts{22.0f, 0.0f});

  HypothesisStatistic stat_child(5,2, mcts_default_parameters());
  stat_child.update_from_heuristic(heuristic);
  stat_parent.update_statistic(stat_child);

  const auto ucb_stats =stat_parent.get_ucb_statistics();
  const auto node_counts = stat_parent.get_total_node_visits();

  EXPECT_NEAR(ucb_stats.at(1).at(action_idx).action_ego_cost_, 5.3f
                    +mcts_default_parameters().DISCOUNT_FACTOR*22.0f, 0.001);
  EXPECT_EQ(ucb_stats.at(1).at(action_idx).action_count_, 1);
  EXPECT_EQ(node_counts.at(1), 1);
}

TEST(hypothesis_statistic, worst_case_action_selection) {
  // Now test something for agent 2
  const std::unordered_map<AgentIdx, HypothesisId> current_agents_hypothesis = {
      {1,0}, {2,1}
  };

  auto mcts_params = mcts_default_parameters();
  mcts_params.hypothesis_statistic.COST_BASED_ACTION_SELECTION = true;
  mcts_params.hypothesis_statistic.PROGRESSIVE_WIDENING_ALPHA = 0.9;
  mcts_params.hypothesis_statistic.PROGRESSIVE_WIDENING_K = 1;

  HypothesisStatisticTestState state(current_agents_hypothesis);
  HypothesisStatistic stat_parent(2,2, mcts_params); // agents 2 statistic 
  auto action_idx = stat_parent.choose_next_action(state);
  stat_parent.collect( 1, EgoCosts{5.3f, 0.0}, action_idx, {0,0});

  HypothesisStatistic heuristic(2,2, mcts_params);
  heuristic.set_heuristic_estimate(10.0f , EgoCosts{22.0f, 0.0f});

  HypothesisStatistic stat_child(2,2, mcts_params);
  stat_child.update_from_heuristic(heuristic);
  stat_parent.update_statistic(stat_child);

  state.change_actions();

  auto action_idx2 = stat_parent.choose_next_action(state);
  stat_parent.collect( 1, EgoCosts{12.3f, 0.0}, action_idx, {0,0});
  stat_parent.update_statistic(stat_child);

  // Worst case action
  auto action_worst_case = stat_parent.choose_next_action(state);
  EXPECT_EQ(action_worst_case, action_idx2);

}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();

}

