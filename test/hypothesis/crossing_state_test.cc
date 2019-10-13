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
#include "test/hypothesis/crossing_state.h"
#include "mcts/hypothesis/hypothesis_belief_tracker.h"
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
    belief_tracker.belief_update(*state);

    std::vector<Reward> rewards;
    Cost cost;
    bool collision = false;

    // All agents move forward 
    const auto action = JointAction(state->get_agent_idx().size(), aconv(1));
    for(int i = 0; i< 100; ++i) {
      state = state->execute(action, rewards, cost);
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
    belief_tracker.belief_update(*state);

    std::vector<Reward> rewards;
    Cost cost;
    bool collision = false;

    // Ego agent moves forward other agents stick to deterministic hypothesis keeping distance of 5
    for(int i = 0; i< 100; ++i) {
      belief_tracker.sample_current_hypothesis();
      auto jointaction = JointAction(state->get_agent_idx().size());
      for (auto agent_idx : state->get_agent_idx()) {
        if (agent_idx == CrossingState::ego_agent_idx ) {
          jointaction[agent_idx] = aconv(1);
        } else {
          const auto action = state->plan_action_current_hypothesis(agent_idx);
          jointaction[agent_idx] = action;
        }
      }
      const auto action_other = 
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
    belief_tracker.belief_update(*state);

    AgentPolicyCrossingState true_agents_policy({5,5});

    std::vector<Reward> rewards;
    Cost cost;
    bool collision = false;

    for(int i = 0; i< 100; ++i) {
      belief_tracker.sample_current_hypothesis();
      auto jointaction = JointAction(state->get_agent_idx().size());
      for (auto agent_idx : state->get_agent_idx()) {
        if (agent_idx == CrossingState::ego_agent_idx ) {
          jointaction[agent_idx] =  aconv(1);
        } else {
          const auto action = true_agents_policy.act(state->get_agent_state(agent_idx),
                                                     state->get_ego_state().x_pos);
          jointaction[agent_idx] = aconv(action);
        }
      }
      std::cout << "Step " << i << ", Action = " << jointaction << ", " << state->sprintf() << std::endl;
      state = state->execute(jointaction, rewards, cost);
      if (state->is_terminal()) {
        break;
      }
      belief_tracker.belief_update(*state);
    }

    const auto beliefs = belief_tracker.get_beliefs();

    // Both beliefs should be equal as they cover the same amount of true policy behavior space
    EXPECT_EQ(beliefs.at(1)[0], beliefs.at(1)[1]);
}


TEST(crossing_state, mcts_goal_reached)
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
    //state->add_hypothesis(AgentPolicyCrossingState({4,5}));
    state->add_hypothesis(AgentPolicyCrossingState({5,5}));
    belief_tracker.belief_update(*state);

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
          std::cout << "best uct action: " << jointaction[agent_idx] << std::endl;
        } else {
          // Other agents act according to unknown true agents policy
          const auto action = true_agents_policy.act(state->get_agent_state(agent_idx),
                                                     state->get_ego_state().x_pos);
          jointaction[agent_idx] = aconv(action);
        }
      }
      std::cout << "Step " << i << ", Action = " << jointaction << ", " << state->sprintf() << std::endl;
      state = state->execute(jointaction, rewards, cost);
      if (state->is_terminal()) {
        break;
      }
      belief_tracker.belief_update(*state);
    }
    EXPECT_TRUE(state->ego_goal_reached());

}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();

}