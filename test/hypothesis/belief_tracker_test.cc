// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================

#include "gtest/gtest.h"

#define UNIT_TESTING
#define DEBUG
#define PLAN_DEBUG_INFO
#include "mcts/mcts_parameters.h"
#include "mcts/hypothesis/hypothesis_statistic.h"
#include "mcts/statistics/uct_statistic.h"
#include "test/hypothesis/belief_tracker_test_state.h"
#include "mcts/hypothesis/hypothesis_belief_tracker.h"
#include <cstdio>

using namespace std;
using namespace mcts;


TEST(belief_tracker, simple_tracking_state)
{
    HypothesisBeliefTracker tracker(mcts_default_parameters());

    // Inits reference to current sampled hypothesis
    BeliefTrackerTestState state(tracker.sample_current_hypothesis()); 
    tracker.belief_update(state, state); //< last action equal in both states
    auto beliefs = tracker.get_beliefs();
    EXPECT_NEAR(beliefs[0][0], 0.5*0.3/(0.5*0.3 +  0.7*0.7), 0.001); // prior x prob(last_action) / normalize by total agent belief
    EXPECT_NEAR(beliefs[1][0], 0.6*0.2/( 0.6*0.2 + 0.4*0.4), 0.001); //  -- "" --
    EXPECT_NEAR(beliefs[0][1], 0.7*0.7/(0.5*0.3 +  0.7*0.7), 0.001); //  -- "" --
    EXPECT_NEAR(beliefs[1][1], 0.4*0.4/( 0.6*0.2 + 0.4*0.4), 0.001); //  -- "" --

    const auto& sampled_hypothesis = tracker.sample_current_hypothesis();

    tracker.belief_update(state, state); //< last action equal in both states
    beliefs = tracker.get_beliefs();
    EXPECT_NEAR(beliefs[0][0], 0.5*0.3*0.3/(0.5*0.3*0.3 +  0.7*0.7*0.7), 0.001); // prior x prob(last_action)
    EXPECT_NEAR(beliefs[1][0], 0.6*0.2*0.2/( 0.6*0.2*0.2 + 0.4*0.4*0.4), 0.001); //  -- "" --
    EXPECT_NEAR(beliefs[0][1], 0.7*0.7*0.7/(0.5*0.3*0.3 +  0.7*0.7*0.7), 0.001); //  -- "" --
    EXPECT_NEAR(beliefs[1][1], 0.4*0.4*0.4/( 0.6*0.2*0.2 + 0.4*0.4*0.4), 0.001); //  -- "" --

    // Check if hypothesis are accurately sampled from belief
    beliefs = tracker.get_beliefs();
    std::unordered_map<AgentIdx, std::unordered_map<HypothesisId,uint>> counts;

    const uint num_samples = 10000;
    for(uint i = 0; i < num_samples; ++i) {
      const auto& sampled = tracker.sample_current_hypothesis();
      for (auto agent_it : sampled) {
        auto& count_agent =  counts[agent_it.first];
        count_agent[sampled.at(agent_it.first)]++;
      }
    }

    EXPECT_NEAR(counts[0][0]/float(num_samples), beliefs[0][0], 0.01);
    EXPECT_NEAR(counts[1][0]/float(num_samples), beliefs[1][0], 0.01);
    EXPECT_NEAR(counts[0][1]/float(num_samples), beliefs[0][1], 0.01);
    EXPECT_NEAR(counts[1][1]/float(num_samples), beliefs[1][1], 0.01);
    
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();

}