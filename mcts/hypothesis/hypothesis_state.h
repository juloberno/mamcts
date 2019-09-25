// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================

#ifndef MCTS_HYPOTHESIS_STATE_H
#define MCTS_HYPOTHESIS_STATE_H

#include "mcts/mcts.h"


namespace mcts {

template<typename Implementation>
class HypothesisState  {
public:
    HypothesisState(const std::vector<const Hypothesis>& hypothesis_available) : hypothesis_available_(hypothesis_available) {}
    void sample_hypothesis_each_agent(); // samples from the probability distribution of hypothesis a new hypothesis for each agent
                                         // and assigns it to the current_agent_hypothesis

    static std::vector<const Hypothesis> define_hypothesis(); // define the hypothesis used for one search, call before search starts
private:
    const std::unordered_map<AgentIdx, const Hypothesis* const> current_agent_hypothesis_; // shared across all states
    const& std::vector<const Hypothesis> hypothesis_available_;
    const std::vector<float> hypothesis_probabilities_;
};

} // namespace mcts

#endif // MCTS_HYPOTHESIS_STATE_H