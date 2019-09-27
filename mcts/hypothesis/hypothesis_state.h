// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================

#ifndef MCTS_HYPOTHESIS_STATE_H
#define MCTS_HYPOTHESIS_STATE_H

#include "mcts/mcts.h"


namespace mcts {

typedef unsigned int HypothesisId;

template<typename Implementation, typename ActionType = ActionIdx>
class HypothesisState : StateInterface<Implementation>  {
public:
    HypothesisState(const std::vector<const Hypothesis>& hypothesis_available,
                    const std::unordered_map<AgentIdx, HypothesisId>& current_agents_hypothesis) 
                    : hypothesis_available_(hypothesis_available),
                      current_agents_hypothesis_(current_agents_hypothesis) {}

    ActionIdx plan_action_current_hypothesis(const AgentIdx& agent_idx) const;

    Probability get_probability(const HypothesisId& hypothesis, const ActionType& action) const;

    Probability get_prior(const HypothesisId& hypothesis) const;

    HypothesisId get_num_hypothesis() const;

private:
    const std::unordered_map<AgentIdx, HypothesisId>& current_agent_hypothesis_; // shared across all states
};

} // namespace mcts

#endif // MCTS_HYPOTHESIS_STATE_H