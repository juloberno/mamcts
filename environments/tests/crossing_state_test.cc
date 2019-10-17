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


std::mt19937  mcts::RandomGenerator::random_generator_;


TEST(hypothesis_crossing_state, collision )
{
    RandomGenerator::random_generator_ = std::mt19937(1000);
    HypothesisBeliefTracker belief_tracker(100, 1, HypothesisBeliefTracker::PRODUCT);
    auto state = std::make_shared<CrossingState>(belief_tracker.sample_current_hypothesis());
    state->add_hypothesis(AgentPolicyCrossingState({5,5}));
    belief_tracker.belief_update(*state, *state);

    std::vector<Reward> rewards;
    Cost cost;
    bool collision = false;

    // All agents move forward 
    auto jointaction = JointAction(state->get_agent_idx().size());
    for (auto agent_idx : state->get_agent_idx()) {
      if (agent_idx == CrossingState::ego_agent_idx ) {
        jointaction[agent_idx] = 2;
      } else {
        const auto action = aconv(1);
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

TEST(hypothesis_crossing_state, hypothesis_friendly)
{
    RandomGenerator::random_generator_ = std::mt19937(1000);
    HypothesisBeliefTracker belief_tracker(100, 1, HypothesisBeliefTracker::PRODUCT);
    auto state = std::make_shared<CrossingState>(belief_tracker.sample_current_hypothesis());
    state->add_hypothesis(AgentPolicyCrossingState({5,5}));
    belief_tracker.belief_update(*state, *state);

    std::vector<Reward> rewards;
    Cost cost;
    bool collision = false;

    // Ego agent moves forward other agents stick to deterministic hypothesis keeping distance of 5
    for(int i = 0; i< 100; ++i) {
      belief_tracker.sample_current_hypothesis();
      auto jointaction = JointAction(state->get_agent_idx().size());
      for (auto agent_idx : state->get_agent_idx()) {
        if (agent_idx == CrossingState::ego_agent_idx ) {
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
    EXPECT_EQ(state->min_distance_to_ego(), 6); // the deterministic hypothesis keeps a desired gap of 5
    EXPECT_FALSE(collision);
}

TEST(hypothesis_crossing_state, hypothesis_belief_correct)
{ 
    // This test checks if hypothesis probability is split up correctly between two overlapping hypothesis
    RandomGenerator::random_generator_ = std::mt19937(1000);
    HypothesisBeliefTracker belief_tracker(4, 1, HypothesisBeliefTracker::PRODUCT);
    auto state = std::make_shared<CrossingState>(belief_tracker.sample_current_hypothesis());
    state->add_hypothesis(AgentPolicyCrossingState({4,5}));
    state->add_hypothesis(AgentPolicyCrossingState({5,6}));
    auto next_state = state;

    AgentPolicyCrossingState true_agents_policy({5,5});

    std::vector<Reward> rewards;
    Cost cost;
    bool collision = false;

    for(int i = 0; i< 100; ++i) {
      belief_tracker.sample_current_hypothesis();
      auto jointaction = JointAction(state->get_agent_idx().size());
      for (auto agent_idx : state->get_agent_idx()) {
        if (agent_idx == CrossingState::ego_agent_idx ) {
          jointaction[agent_idx] =  2;
        } else {
          const auto action = true_agents_policy.act(state->get_agent_state(agent_idx),
                                                     state->get_ego_state().x_pos);
          jointaction[agent_idx] = aconv(action);
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
    EXPECT_EQ(beliefs.at(1)[0], beliefs.at(1)[1]);
}


TEST(crossing_state, mcts_goal_reached_true_hypothesis)
{
    MctsParameters::DISCOUNT_FACTOR = 0.9;
    MctsParameters::RandomHeuristic::MAX_SEARCH_TIME = 10;
    MctsParameters::RandomHeuristic::MAX_NUMBER_OF_ITERATIONS = 1000;
    MctsParameters::UctStatistic::LOWER_BOUND = -1010;
    MctsParameters::UctStatistic::UPPER_BOUND = 95;
    MctsParameters::UctStatistic::EXPLORATION_CONSTANT = 0.7;
    MctsParameters::HypothesisStatistic::COST_BASED_ACTION_SELECTION = true;
    MctsParameters::HypothesisStatistic::LOWER_COST_BOUND = 0;
    MctsParameters::HypothesisStatistic::UPPER_COST_BOUND = 1;
    MctsParameters::HypothesisStatistic::PROGRESSIVE_WIDENING_ALPHA = 0.5;
    MctsParameters::HypothesisStatistic::PROGRESSIVE_WIDENING_K = 1;
    MctsParameters::HypothesisStatistic::EXPLORATION_CONSTANT = 0.7;


    RandomGenerator::random_generator_ = std::mt19937(1000);
    HypothesisBeliefTracker belief_tracker(4, 1, HypothesisBeliefTracker::PRODUCT);
    auto state = std::make_shared<CrossingState>(belief_tracker.sample_current_hypothesis());
    state->add_hypothesis(AgentPolicyCrossingState({5,5}));
    auto next_state = state;
    belief_tracker.belief_update(*state, *next_state);

    AgentPolicyCrossingState true_agents_policy({5,5});

    std::vector<Reward> rewards;
    Cost cost;
    bool collision = false;

    for(int i = 0; i< 100; ++i) {
      auto jointaction = JointAction(state->get_agent_idx().size());
      for (auto agent_idx : state->get_agent_idx()) {
        if (agent_idx == CrossingState::ego_agent_idx ) {
          // Plan for ego agent with hypothesis-based search
          Mcts<CrossingState, UctStatistic, HypothesisStatistic, RandomHeuristic> mcts;
          mcts.search(*state, belief_tracker, 200, 2000);
          jointaction[agent_idx] = mcts.returnBestAction();
          std::cout << "best uct action: " << idx_to_ego_crossing_action(jointaction[agent_idx]) << std::endl;
        } else {
          // Other agents act according to unknown true agents policy
          const auto action = true_agents_policy.act(state->get_agent_state(agent_idx),
                                                     state->get_ego_state().x_pos);
          jointaction[agent_idx] = aconv(action);
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
    MctsParameters::DISCOUNT_FACTOR = 0.9;
    MctsParameters::RandomHeuristic::MAX_SEARCH_TIME = 10;
    MctsParameters::RandomHeuristic::MAX_NUMBER_OF_ITERATIONS = 1000;
    MctsParameters::UctStatistic::LOWER_BOUND = -1010;
    MctsParameters::UctStatistic::UPPER_BOUND = 95;
    MctsParameters::UctStatistic::EXPLORATION_CONSTANT = 0.7;
    MctsParameters::HypothesisStatistic::COST_BASED_ACTION_SELECTION = false;
    MctsParameters::HypothesisStatistic::LOWER_COST_BOUND = 0;
    MctsParameters::HypothesisStatistic::UPPER_COST_BOUND = 1;
    MctsParameters::HypothesisStatistic::PROGRESSIVE_WIDENING_ALPHA = 0.5;
    MctsParameters::HypothesisStatistic::PROGRESSIVE_WIDENING_K = 1;
    MctsParameters::HypothesisStatistic::EXPLORATION_CONSTANT = 0.7;


    RandomGenerator::random_generator_ = std::mt19937(1000);
    HypothesisBeliefTracker belief_tracker(4, 1, HypothesisBeliefTracker::PRODUCT);
    auto state = std::make_shared<CrossingState>(belief_tracker.sample_current_hypothesis());
    state->add_hypothesis(AgentPolicyCrossingState({-2,-1}));
    state->add_hypothesis(AgentPolicyCrossingState({0,3}));
    state->add_hypothesis(AgentPolicyCrossingState({4,5}));
    auto next_state = state;
    belief_tracker.belief_update(*state, *next_state);

    AgentPolicyCrossingState true_agents_policy({-2,-2});

    std::vector<Reward> rewards;
    Cost cost;
    bool collision = false;

    for(int i = 0; i< 30; ++i) {
      auto jointaction = JointAction(state->get_agent_idx().size());
      for (auto agent_idx : state->get_agent_idx()) {
        if (agent_idx == CrossingState::ego_agent_idx ) {
          // Plan for ego agent with hypothesis-based search
          Mcts<CrossingState, UctStatistic, HypothesisStatistic, RandomHeuristic> mcts;
          mcts.search(*state, belief_tracker, 5000, 10000);
          jointaction[agent_idx] = mcts.returnBestAction();
          std::cout << "best uct action: " << idx_to_ego_crossing_action(jointaction[agent_idx]) << std::endl;
        } else {
          // Other agents act according to unknown true agents policy
          const auto action = true_agents_policy.act(state->get_agent_state(agent_idx),
                                                     state->get_ego_state().x_pos);
          jointaction[agent_idx] = aconv(action);
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
  CrossingStateParameters::CHAIN_LENGTH = 3;
  CrossingStateParameters::EGO_GOAL_POS = 1;
  auto runner = CrossingStateEpisodeRunner(
      { {1 , AgentPolicyCrossingState({5,5})},
        {2 , AgentPolicyCrossingState({5,5})}},
      {AgentPolicyCrossingState({4,5}), 
        AgentPolicyCrossingState({5,6})},
        30,
        nullptr);
  for (int i =0; i< 50; ++i) {
    runner.step();
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();

}