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

using Domain = int;


TEST(hypothesis_crossing_state, collision )
{ 
    const auto params = default_crossing_state_parameters<Domain>();
    auto mcts_params =mcts_default_parameters();
    mcts_params.hypothesis_belief_tracker.HISTORY_LENGTH = 100;
    HypothesisBeliefTracker belief_tracker(mcts_params);
    auto state = std::make_shared<CrossingState<Domain>>(belief_tracker.sample_current_hypothesis(), params);
    state->add_hypothesis(AgentPolicyCrossingState<Domain>({5,5}, params));
    belief_tracker.belief_update(*state, *state);

    std::vector<Reward> rewards;
    EgoCosts cost;
    bool collision = false;

    // All agents move forward 
    auto jointaction = JointAction(state->get_num_agents());
    jointaction[CrossingState<Domain>::ego_agent_idx] =2;
    for (auto agent_idx : state->get_other_agent_idx()) {
      const auto action = aconv<Domain>(1);
      jointaction[agent_idx] = action;
    }
    for(int i = 0; i< 100; ++i) {
      state = state->execute(jointaction, rewards, cost);
      if (state->is_terminal() && !state->ego_goal_reached()) {
        collision=true;
        break;
      }
    }
    EXPECT_TRUE(collision);
}

TEST(hypothesis_crossing_state, hypothesis_friendly)
{
    const auto params = default_crossing_state_parameters<Domain>();
    auto mcts_params =mcts_default_parameters();
    mcts_params.hypothesis_belief_tracker.HISTORY_LENGTH = 100;
    mcts_params.NUM_PARALLEL_MCTS = 4;
    HypothesisBeliefTracker belief_tracker(mcts_params);
    auto state = std::make_shared<CrossingState<Domain>>(belief_tracker.sample_current_hypothesis(), params);
    state->add_hypothesis(AgentPolicyCrossingState<Domain>({5,5}, params));
    belief_tracker.belief_update(*state, *state);

    std::vector<Reward> rewards;
    EgoCosts cost;
    bool collision = false;

    // Ego agent moves forward other agents stick to deterministic hypothesis keeping distance of 5
    for(int i = 0; i< 100; ++i) {
      belief_tracker.sample_current_hypothesis();
      auto jointaction = JointAction(state->get_num_agents());
      jointaction[CrossingState<Domain>::ego_agent_idx] = 2;
      for (auto agent_idx : state->get_other_agent_idx()) {
        const auto action = state->plan_action_current_hypothesis(agent_idx);
        jointaction[agent_idx] = action;
      }
      state = state->execute(jointaction, rewards, cost);
      if (state->is_terminal() && !state->ego_goal_reached()) {
        collision=true;
        break;
      }
      if(state->is_terminal()) {
        break;
      }
    }
    EXPECT_EQ(state->min_distance_to_ego(), 5); // the deterministic hypothesis keeps a desired gap of 5
    EXPECT_FALSE(collision);
}

TEST(hypothesis_crossing_state, hypothesis_belief_correct)
{  
    // This test checks if hypothesis probability is split up correctly between two overlapping hypothesis
    const auto params = default_crossing_state_parameters<Domain>();
    auto mcts_params =mcts_default_parameters();
    mcts_params.hypothesis_belief_tracker.HISTORY_LENGTH = 4;
    HypothesisBeliefTracker belief_tracker(mcts_params);
    auto state = std::make_shared<CrossingState<Domain>>(belief_tracker.sample_current_hypothesis(), params);
    state->add_hypothesis(AgentPolicyCrossingState<Domain>({4,5}, params));
    state->add_hypothesis(AgentPolicyCrossingState<Domain>({5,6}, params));
    auto next_state = state;

    AgentPolicyCrossingState<Domain> true_agents_policy({5,5}, params);

    std::vector<Reward> rewards;
    EgoCosts cost;
    bool collision = false;

    for(int i = 0; i< 100; ++i) {
      belief_tracker.sample_current_hypothesis();
      auto jointaction = JointAction(state->get_num_agents());
      jointaction[CrossingState<Domain>::ego_agent_idx] =  2;
      for (auto agent_idx : state->get_other_agent_idx()) {
        const auto action = true_agents_policy.act(state->get_agent_state(agent_idx),
                                                    state->get_ego_state());
        jointaction[agent_idx] = aconv<Domain>(action);
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
    const auto params = default_crossing_state_parameters<Domain>();
    auto mcts_params =mcts_default_parameters();
    mcts_params.hypothesis_belief_tracker.HISTORY_LENGTH = 4;
    HypothesisBeliefTracker belief_tracker(mcts_params);
    auto state = std::make_shared<CrossingState<Domain>>(belief_tracker.sample_current_hypothesis(), params);
    state->add_hypothesis(AgentPolicyCrossingState<Domain>({5,5}, params));
    auto next_state = state;
    belief_tracker.belief_update(*state, *next_state);

    AgentPolicyCrossingState<Domain> true_agents_policy({5,5}, params);

    std::vector<Reward> rewards;
    EgoCosts cost;
    bool collision = false;

    for(int i = 0; i< 100; ++i) {
      auto jointaction = JointAction(state->get_num_agents());
      // Plan for ego agent with hypothesis-based search
      Mcts<CrossingState<Domain>, UctStatistic, HypothesisStatistic, RandomHeuristic> mcts(mcts_default_parameters());
      mcts.search(*state, belief_tracker);
      jointaction[CrossingState<Domain>::ego_agent_idx] = mcts.returnBestAction();
      std::cout << "best uct action: " <<
       state->idx_to_ego_crossing_action(jointaction[CrossingState<Domain>::ego_agent_idx]) << std::endl;
      for (auto agent_idx : state->get_other_agent_idx()) {
        // Other agents act according to unknown true agents policy
        const auto action = true_agents_policy.act(state->get_agent_state(agent_idx),
                                                    state->get_ego_state());
        jointaction[agent_idx] = aconv<Domain>(action);
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
    const auto params = default_crossing_state_parameters<Domain>();
    auto mcts_params =mcts_default_parameters();
    HypothesisBeliefTracker belief_tracker(mcts_params);
    auto state = std::make_shared<CrossingState<Domain>>(belief_tracker.sample_current_hypothesis(), params);
    state->add_hypothesis(AgentPolicyCrossingState<Domain>({-2,1}, params));
    state->add_hypothesis(AgentPolicyCrossingState<Domain>({2,3}, params));
    state->add_hypothesis(AgentPolicyCrossingState<Domain>({4,5}, params));
    auto next_state = state;
    belief_tracker.belief_update(*state, *next_state);

    AgentPolicyCrossingState<Domain> true_agents_policy({-2,-2}, params);

    std::vector<Reward> rewards;
    EgoCosts cost;
    bool collision = false;

    for(int i = 0; i< 30; ++i) {
      auto jointaction = JointAction(state->get_num_agents());
      Mcts<CrossingState<Domain>, UctStatistic, HypothesisStatistic, RandomHeuristic> mcts(mcts_default_parameters());
      mcts.search(*state, belief_tracker);
      jointaction[CrossingState<Domain>::ego_agent_idx] = mcts.returnBestAction();
      std::cout << "best uct action: " << state->idx_to_ego_crossing_action(jointaction[CrossingState<Domain>::ego_agent_idx]) << std::endl;
      for (auto agent_idx : state->get_other_agent_idx()) {
        // Other agents act according to unknown true agents policy
        const auto action = true_agents_policy.act(state->get_agent_state(agent_idx),
                                                    state->get_ego_state());
        jointaction[agent_idx] = aconv<Domain>(action);
      }
      std::cout << "Step " << i << ", Action = " << jointaction << ", " << state->sprintf() << std::endl;
      next_state = state->execute(jointaction, rewards, cost);
      belief_tracker.belief_update(*state, *next_state);
      state = next_state;
      if (state->is_terminal() && !state->ego_goal_reached()) {
        collision=true;
        break;
      }
      if (next_state->is_terminal()) {
        break;
      }
    }
    EXPECT_TRUE(state->ego_goal_reached());
    EXPECT_FALSE(collision);

}

TEST(episode_runner, four_agents_reached_goal) {
  auto params = default_crossing_state_parameters<Domain>();
  params.NUM_OTHER_AGENTS = 4;
  params.CHAIN_LENGTH = 41;
  params.EGO_GOAL_POS = 26;
  auto runner = CrossingStateEpisodeRunner<Domain>(
      { {1 , AgentPolicyCrossingState<Domain>({5,5}, params)},
        {2 , AgentPolicyCrossingState<Domain>({4,4}, params)},
        {3 , AgentPolicyCrossingState<Domain>({6,6}, params)},
        {4 , AgentPolicyCrossingState<Domain>({-2,-2}, params)}},
      {AgentPolicyCrossingState<Domain>({4,5}, params), 
        AgentPolicyCrossingState<Domain>({-2,3}, params),
        AgentPolicyCrossingState<Domain>({5,6}, params)},
        mcts_default_parameters(),
        params,
        30,
        200,
        10000,
        nullptr);
  auto result = runner.run();
  EXPECT_TRUE(std::get<4>(result).second);
}

TEST(episode_runner, run_some_steps) {
  auto params = default_crossing_state_parameters<Domain>();
  params.CHAIN_LENGTH = 3;
  params.EGO_GOAL_POS = 1;
  auto runner = CrossingStateEpisodeRunner<Domain>(
      { {1 , AgentPolicyCrossingState<Domain>({5,5}, params)},
        {2 , AgentPolicyCrossingState<Domain>({5,5}, params)}},
      {AgentPolicyCrossingState<Domain>({4,5}, params), 
        AgentPolicyCrossingState<Domain>({5,6}, params)},
        mcts_default_parameters(),
        params,
        30,
        200,
        10000,
        nullptr);
  for (int i =0; i< 50; ++i) {
    runner.step();
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();

}