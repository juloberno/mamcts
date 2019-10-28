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
#include "mcts/hypothesis/hypothesis_belief_tracker.h"

#include "environments/crossing_state.h"
#include "environments/crossing_state_episode_runner.h"

#include <cstdio>

using namespace std;
using namespace mcts;

MctsParameters default_hypo_params() {
  MctsParameters parameters;
  parameters.DISCOUNT_FACTOR = 0.9;
  
  parameters.random_heuristic.MAX_SEARCH_TIME = 10;
  parameters.random_heuristic.MAX_NUMBER_OF_ITERATIONS = 1000;

  parameters.uct_statistic.LOWER_BOUND = -1000;
  parameters.uct_statistic.UPPER_BOUND = 100;
  parameters.uct_statistic.EXPLORATION_CONSTANT = 0.7;

  parameters.hypothesis_statistic.COST_BASED_ACTION_SELECTION = false;
  parameters.hypothesis_statistic.LOWER_COST_BOUND = 0;
  parameters.hypothesis_statistic.UPPER_COST_BOUND = 1;
  parameters.hypothesis_statistic.PROGRESSIVE_WIDENING_ALPHA = 0.5;
  parameters.hypothesis_statistic.PROGRESSIVE_WIDENING_K = 1;
  parameters.hypothesis_statistic.EXPLORATION_CONSTANT = 0.7;

  return parameters;
}


std::mt19937  mcts::RandomGenerator::random_generator_;

using Domain = float;

Probability policy_test_helper(const float& agent_pos, const float& agent_last_action,
                               const float& ego_pos, std::pair<float,float> gap_range,
                               const float& action) {
  AgentPolicyCrossingState<Domain> policy(gap_range);
  AgentState<Domain> astate(agent_pos, agent_last_action);
  const auto p = policy.get_probability(astate, ego_pos, action);
  return p;
}


TEST(hypothesis_crossing_state_float, policy_probability )
{
  CrossingStateParameters<float>::MAX_VELOCITY_OTHER = 3;
  CrossingStateParameters<float>::MIN_VELOCITY_OTHER = -3;

  // Both gap bounds yield negative gap error
  auto p = policy_test_helper(1, 2, 1, {2, 3.5}, -2.5);
  EXPECT_NEAR(p, 1/(3.5f-2.0f)*0.001, 0.001f);

  p = policy_test_helper(1, 2, 1.8, {2, 3.5}, -2.5);
  EXPECT_NEAR(p, 1/(3.5f-2.0f)*0.001, 0.001f);

  p = policy_test_helper(1, 2, 2, {4.5, 5}, -3);
  EXPECT_NEAR(p, 1.0f, 0.001f);

  p = policy_test_helper(1, 2, 1.7f, {3, 5}, -3);
  EXPECT_NEAR(p, 1.3/(5-3) , 0.001f);

  p = policy_test_helper(1, 2, 3.0f, {3, 5}, -3);
  EXPECT_NEAR(p, 1.0/(5-3)*0.001 , 0.001f);

  p = policy_test_helper(1, 2, 0.0f, {3, 5}, -4.5);
  EXPECT_NEAR(p, 1.0/(5-3)*0.001 , 0.001f);

  p = policy_test_helper(1, 2, 0.0f, {0, 5}, -3.0f);
  EXPECT_NEAR(p, 3.0/(5-0) , 0.001f);

  // One gap bound yields positive, one gap bound negative error
  p = policy_test_helper(1, 2, 2.5f, {1, 5}, 0.3f);
  EXPECT_NEAR(p, 1.0/(5.0-1.0)*0.001 , 0.001f);

  p = policy_test_helper(1, 2, 2.5f, {1, 5}, 0.5f);
  EXPECT_NEAR(p, 1.0/(5.0-1.0)*0.001 , 0.001f);

  p = policy_test_helper(1, 2, 2.5f, {1, 5}, 1.5f);
  EXPECT_NEAR(p, 0.0f , 0.001f);

  p = policy_test_helper(1, 2, 2.5f, {1, 5}, -3.0f);
  EXPECT_NEAR(p, 0.5/(5.0-1.0), 0.001f);

  p = policy_test_helper(1, 2, 2.5f, {1, 4.5}, -3.0f);
  EXPECT_NEAR(p, 1.0f/(4.5-1.0)*0.001 , 0.001f);

  // Both gap bounds yield positive error

  // -- last action not used -------
  p = policy_test_helper(1, 1, 0.5f, {-3.0f, -2.5f}, 1.5f);
  EXPECT_NEAR(p, 0.0f , 0.001f);

  p = policy_test_helper(1, 1, 0.5f, {-3.0f, -2.5f}, 2.5f);
  EXPECT_NEAR(p, 1.0f/(-2.5+3)*0.001 , 0.001f);

  p = policy_test_helper(1, 1, 0.5f, {-4.0f, -3.5f}, 3.0f);
  EXPECT_NEAR(p, 1.0f/(4-3.5)*0.001 , 0.001f);

  p = policy_test_helper(1, 1, 0.5f, {-4.0f, -3.5f}, 2.5f);
  EXPECT_NEAR(p, 0.0f , 0.001f);

  p = policy_test_helper(1, 1, 0.5f, {-4.0f, -3.5f}, 5.5f);
  EXPECT_NEAR(p, 0.0f , 0.001f);

  p = policy_test_helper(1, 1, 0.5f, {-3.0f, -2.5f}, 5.5f);
  EXPECT_NEAR(p, 0.0f , 0.001f);

  p = policy_test_helper(1, 2.5, 0.5f, {-5.0f, -2.5f}, 3.0f);
  EXPECT_NEAR(p, 1.5/(5-2.5) , 0.001f);

  // -- last action used---
  p = policy_test_helper(1, 1, 0.5f, {-3.0f, -2.5f}, 1.0f);
  EXPECT_NEAR(p, 0.0f , 0.001f);

  p = policy_test_helper(1, 3.5, 0.5f, {-3.0f, -2.5f}, 2.5f);
  EXPECT_NEAR(p, 0.0f , 0.001f);

  p = policy_test_helper(1, 3.5, 0.5f, {-3.0f, -2.5f}, 2.5f);
  EXPECT_NEAR(p, 0.0f , 0.001f);

  p = policy_test_helper(1, 2.5, 0.5f, {-5.0f, -2.5f}, 2.8f);
  EXPECT_NEAR(p, 1.0/(5-2.5)*0.001 , 0.001f);

  p = policy_test_helper(1, 2.5, 0.5f, {-5.0f, -4.5f}, 3.0f);
  EXPECT_NEAR(p, 1.0, 0.001f);

  p = policy_test_helper(1, 1.5, 0.5f, {-5.0f, -1.5f}, 1.5f);
  EXPECT_NEAR(p, 1.0/(5-1.5)*0.001, 0.001f);

  p = policy_test_helper(1, 3.5, 0.5f, {-1.5f, -0.5f}, 3.5f);
  EXPECT_NEAR(p, 1.0, 0.001f);
}

TEST(hypothesis_crossing_state_float, collision )
{
    RandomGenerator::random_generator_ = std::mt19937(1000);
    HypothesisBeliefTracker belief_tracker(100, 1, HypothesisBeliefTracker::PRODUCT);
    auto state = std::make_shared<CrossingState<Domain>>(belief_tracker.sample_current_hypothesis());
    state->add_hypothesis(AgentPolicyCrossingState<Domain>({5.0f, 5.5f}));
    belief_tracker.belief_update(*state, *state);

    std::vector<Reward> rewards;
    Cost cost;
    bool collision = false;

    // All agents move forward 
    auto jointaction = JointAction(state->get_agent_idx().size());
    for (auto agent_idx : state->get_agent_idx()) {
      if (agent_idx == CrossingState<Domain>::ego_agent_idx ) {
        jointaction[agent_idx] = 2;
      } else {
        const auto action = aconv<Domain>(1.0f);
        jointaction[agent_idx] = action;
      }
    }
    for(int i = 0; i< 100; ++i) {
      state = state->execute(jointaction, rewards, cost);
      if (cost > 0 && state->is_terminal()) {
        collision = true;
        break;
      }
    }
    EXPECT_TRUE(collision);
}

TEST(hypothesis_crossing_state_float, hypothesis_friendly)
{
    RandomGenerator::random_generator_ = std::mt19937(1000);
    HypothesisBeliefTracker belief_tracker(100, 1, HypothesisBeliefTracker::PRODUCT);
    auto state = std::make_shared<CrossingState<Domain>>(belief_tracker.sample_current_hypothesis());
    state->add_hypothesis(AgentPolicyCrossingState<Domain>({5,5.5}));
    belief_tracker.belief_update(*state, *state);

    std::vector<Reward> rewards;
    Cost cost;
    bool collision = false;

    // Ego agent moves forward other agents stick to deterministic hypothesis keeping distance of 5
    for(int i = 0; i< 100; ++i) {
      belief_tracker.sample_current_hypothesis();
      auto jointaction = JointAction(state->get_agent_idx().size());
      for (auto agent_idx : state->get_agent_idx()) {
        if (agent_idx == CrossingState<Domain>::ego_agent_idx ) {
          jointaction[agent_idx] = 2;
        } else {
          const auto action = state->plan_action_current_hypothesis(agent_idx);
          jointaction[agent_idx] = action;
        }
      }
      state = state->execute(jointaction, rewards, cost);
      if (cost > 0) {
        collision=true;
        break;
      }
      if(state->is_terminal()) {
        break;
      }
    }
    EXPECT_TRUE(state->min_distance_to_ego() >= 6); // the deterministic hypothesis keeps a desired gap of 5
    EXPECT_FALSE(collision);
}

TEST(hypothesis_crossing_state_float, hypothesis_belief_correct)
{ 
    CrossingStateParameters<float>::CHAIN_LENGTH = 4000.0;
    CrossingStateParameters<float>::EGO_GOAL_POS = 3999.0;
    // This test checks if hypothesis probability is split up correctly between two overlapping hypothesis
    RandomGenerator::random_generator_ = std::mt19937(1000);
    HypothesisBeliefTracker belief_tracker(200, 1, HypothesisBeliefTracker::SUM);
    auto state = std::make_shared<CrossingState<Domain>>(belief_tracker.sample_current_hypothesis());
    state->add_hypothesis(AgentPolicyCrossingState<Domain>({4,5}));
    state->add_hypothesis(AgentPolicyCrossingState<Domain>({5,6}));
    auto next_state = state;

    AgentPolicyCrossingState<Domain> true_agents_policy({4.5,5.5});

    std::vector<Reward> rewards;
    Cost cost;
    bool collision = false;

    for(int i = 0; i < 200; ++i) {
      belief_tracker.sample_current_hypothesis();
      auto jointaction = JointAction(state->get_agent_idx().size());
      for (auto agent_idx : state->get_agent_idx()) {
        if (agent_idx == CrossingState<Domain>::ego_agent_idx ) {
          jointaction[agent_idx] =  2;
        } else {
          const auto action = true_agents_policy.act(state->get_agent_state(agent_idx),
                                                     state->get_ego_state().x_pos);
          jointaction[agent_idx] = aconv<Domain>(action);
        }
      }
      std::cout << "Step " << i << ", Action = " << jointaction << ", " << state->sprintf() << std::endl;
      next_state = state->execute(jointaction, rewards, cost);
      belief_tracker.belief_update(*state, *next_state);
      state = next_state;
      if (state->is_terminal()) {
        break;
      }
      
    }

    const auto beliefs = belief_tracker.get_beliefs();

    // Both beliefs should be equal as they cover the same amount of true policy behavior space
    EXPECT_NEAR(beliefs.at(1)[0], 0.5, 0.05);
    EXPECT_NEAR(beliefs.at(1)[1], 0.5, 0.05);
}


TEST(crossing_state, mcts_goal_reached_true_hypothesis)
{
    CrossingStateParameters<float>::CHAIN_LENGTH = 41.0;
    CrossingStateParameters<float>::EGO_GOAL_POS = 25.0;

    RandomGenerator::random_generator_ = std::mt19937(1000);
    HypothesisBeliefTracker belief_tracker(10, 1, HypothesisBeliefTracker::SUM);
    auto state = std::make_shared<CrossingState<Domain>>(belief_tracker.sample_current_hypothesis());
    state->add_hypothesis(AgentPolicyCrossingState<Domain>({2,3.5}));
    auto next_state = state;
    belief_tracker.belief_update(*state, *next_state);

    AgentPolicyCrossingState<Domain> true_agents_policy({2,3.5});

    std::vector<Reward> rewards;
    Cost cost;
    bool collision = false;

    for(int i = 0; i< 20; ++i) {
      auto jointaction = JointAction(state->get_agent_idx().size());
      for (auto agent_idx : state->get_agent_idx()) {
        if (agent_idx == CrossingState<Domain>::ego_agent_idx ) {
          // Plan for ego agent with hypothesis-based search
          Mcts<CrossingState<Domain>, UctStatistic, HypothesisStatistic, RandomHeuristic> mcts(default_hypo_params());
          mcts.search(*state, belief_tracker, 200, 10000);
          jointaction[agent_idx] = mcts.returnBestAction();
          std::cout << "best uct action: " << idx_to_ego_crossing_action<Domain>(jointaction[agent_idx]) << ", num iterations: " << mcts.numIterations() << std::endl;
        } else {
          // Other agents act according to unknown true agents policy
          const auto action = true_agents_policy.act(state->get_agent_state(agent_idx),
                                                     state->get_ego_state().x_pos);
          jointaction[agent_idx] = aconv<Domain>(action);
        }
      }
      std::cout << "Step " << i << ", Action = " << jointaction << ", " << state->sprintf() << std::endl;
      next_state = state->execute(jointaction, rewards, cost);
      belief_tracker.belief_update(*state, *next_state);
      state = next_state;
      if (next_state->is_terminal()) {
        break;
      }
    }
    EXPECT_TRUE(state->ego_goal_reached());

}

TEST(crossing_state, mcts_goal_reached_wrong_hypothesis)
{
    CrossingStateParameters<float>::CHAIN_LENGTH = 41.0;
    CrossingStateParameters<float>::EGO_GOAL_POS = 25.0;

    RandomGenerator::random_generator_ = std::mt19937(1000);
    HypothesisBeliefTracker belief_tracker(4, 1, HypothesisBeliefTracker::PRODUCT);
    auto state = std::make_shared<CrossingState<Domain>>(belief_tracker.sample_current_hypothesis());
    state->add_hypothesis(AgentPolicyCrossingState<Domain>({-2,-1}));
    state->add_hypothesis(AgentPolicyCrossingState<Domain>({0,3}));
    state->add_hypothesis(AgentPolicyCrossingState<Domain>({4,5}));
    auto next_state = state;
    belief_tracker.belief_update(*state, *next_state);

    AgentPolicyCrossingState<Domain> true_agents_policy({-2,-1.8});

    std::vector<Reward> rewards;
    Cost cost;
    bool collision = false;

    for(int i = 0; i< 30; ++i) {
      auto jointaction = JointAction(state->get_agent_idx().size());
      for (auto agent_idx : state->get_agent_idx()) {
        if (agent_idx == CrossingState<Domain>::ego_agent_idx ) {
          // Plan for ego agent with hypothesis-based search
          Mcts<CrossingState<Domain>, UctStatistic, HypothesisStatistic, RandomHeuristic> mcts(default_hypo_params());
          mcts.search(*state, belief_tracker, 5000, 10000);
          jointaction[agent_idx] = mcts.returnBestAction();
          std::cout << "best uct action: " << idx_to_ego_crossing_action<Domain>(jointaction[agent_idx]) << std::endl;
        } else {
          // Other agents act according to unknown true agents policy
          const auto action = true_agents_policy.act(state->get_agent_state(agent_idx),
                                                     state->get_ego_state().x_pos);
          jointaction[agent_idx] = aconv<Domain>(action);
        }
      }
      std::cout << "Step " << i << ", Action = " << jointaction << ", " << state->sprintf() << std::endl;
      next_state = state->execute(jointaction, rewards, cost);
      belief_tracker.belief_update(*state, *next_state);
      state = next_state;
      if(cost > 0) {
        collision=true;
      }
      if (next_state->is_terminal()) {
        break;
      }
    }
    EXPECT_TRUE(state->ego_goal_reached());
    EXPECT_FALSE(collision);

}

TEST(episode_runner, run_some_steps) {
  CrossingStateParameters<Domain>::CHAIN_LENGTH = 20;
  CrossingStateParameters<Domain>::EGO_GOAL_POS = 11;
  auto runner = CrossingStateEpisodeRunner<Domain>(
      { {1 , AgentPolicyCrossingState<Domain>({5,5})},
        {2 , AgentPolicyCrossingState<Domain>({5,5})}},
      {AgentPolicyCrossingState<Domain>({4,5}), 
        AgentPolicyCrossingState<Domain>({5,6})},
        30,
        4,
        1,
        HypothesisBeliefTracker::PRODUCT,
        10000,
        10000,
        default_hypo_params(),
        nullptr);
  for (int i =0; i< 50; ++i) {
    runner.step();
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();

}