// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================

#ifndef MCTS_HYPOTHESIS_HYPOTHESISBELIEFTRACKER_H
#define MCTS_HYPOTHESIS_HYPOTHESISBELIEFTRACKER_H

#include "mcts/hypothesis/common.h"
#include "mcts/hypothesis/hypothesis_state.h"

namespace mcts {

template <typename S, typename ActionType>
class HypothesisBeliefTracker {
  public:
    HypothesisBeliefTracker();
    
    void belief_update(HypothesisStateInterface<S> state);

    std::unordered_map<AgentIdx, HypothesisId> sample_current_hypothesis() const; // shared across all states

private:
    std::unordered_map<AgentIdx, std::vector<Belief>> tracked_beliefs_;//< contains the beliefs for each hypothesis for each agent 

};

} // namespace mcts

#endif // MCTS_HYPOTHESIS_HYPOTHESISBELIEFTRACKER_H