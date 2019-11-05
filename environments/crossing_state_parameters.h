// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================

#ifndef CROSSING_STATE_PARAMETERS_H
#define CROSSING_STATE_PARAMETERS_H



namespace mcts {


template <typename Domain>
struct CrossingStateParameters {
    unsigned int NUM_OTHER_AGENTS;
    unsigned int OTHER_AGENTS_POLICY_RANDOM_SEED;
    bool COST_ONLY_COLLISION;
    Domain MAX_VELOCITY_OTHER;
    Domain MIN_VELOCITY_OTHER;
    Domain NUM_OTHER_ACTIONS() const { return MAX_VELOCITY_OTHER-MIN_VELOCITY_OTHER + 1; }
    Domain MAX_VELOCITY_EGO;
    Domain MIN_VELOCITY_EGO;
    Domain NUM_EGO_ACTIONS() const { return MAX_VELOCITY_EGO - MIN_VELOCITY_EGO + 1; }
    Domain CHAIN_LENGTH; 
    Domain EGO_GOAL_POS;
    Domain CROSSING_POINT() const { return (CHAIN_LENGTH-1)/2+1; }
};

template <typename Domain>
CrossingStateParameters<Domain> default_crossing_state_parameters() {
  CrossingStateParameters<Domain> parameters;
  parameters.NUM_OTHER_AGENTS = 2;
  parameters.COST_ONLY_COLLISION = false;
  parameters.OTHER_AGENTS_POLICY_RANDOM_SEED = 1000;
  parameters.MAX_VELOCITY_OTHER = 3;
  parameters.MIN_VELOCITY_OTHER = -3;
  parameters.MAX_VELOCITY_EGO = 2;
  parameters.MIN_VELOCITY_EGO = -1;
  parameters.CHAIN_LENGTH = 21; 
  parameters.EGO_GOAL_POS = 12;

  return parameters;
}

} // namespace mcts

#endif
