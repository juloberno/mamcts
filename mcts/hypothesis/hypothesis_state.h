// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================

#ifndef MCTS_HYPOTHESIS_STATE_H
#define MCTS_HYPOTHESIS_STATE_H

#include "mcts/hypothesis/common.h"
#include "mcts/state.h"


namespace mcts {


template<typename Implementation>
class HypothesisStateInterface : public StateInterface<Implementation>,
                                        mcts::RequiresHypothesis  {
public:
    HypothesisStateInterface(const std::unordered_map<AgentIdx, HypothesisId>& current_agents_hypothesis) 
                    : current_agents_hypothesis_(current_agents_hypothesis) {}

    ActionIdx plan_action_current_hypothesis(const AgentIdx& agent_idx) const;

    template<typename ActionType = ActionIdx>
    Probability get_probability(const HypothesisId& hypothesis, const AgentIdx& agent_idx, const ActionType& action) const;

    template<typename ActionType = ActionIdx>
    ActionType get_last_action(const AgentIdx& agent_idx) const;

    Probability get_prior(const HypothesisId& hypothesis, const AgentIdx& agent_idx) const;

    HypothesisId get_num_hypothesis(const AgentIdx& agent_idx) const;

    HypothesisId get_current_hypothesis(const AgentIdx& agent_idx) const;

    std::shared_ptr<Implementation>  change_belief_reference(const std::unordered_map<AgentIdx, HypothesisId>& current_agents_hypothesis) const;

protected:
    const std::unordered_map<AgentIdx, HypothesisId>& current_agents_hypothesis_; // shared across all states
};

template<typename Implementation>
inline std::shared_ptr<Implementation> HypothesisStateInterface<Implementation>::change_belief_reference(
                const std::unordered_map<AgentIdx, HypothesisId>& current_agents_hypothesis) const {
 return StateInterface<Implementation>::impl().change_belief_reference(current_agents_hypothesis);
}

template<typename Implementation>
inline ActionIdx HypothesisStateInterface<Implementation>::plan_action_current_hypothesis(const AgentIdx& agent_idx) const {
 return StateInterface<Implementation>::impl().plan_action_current_hypothesis(agent_idx);
}

template<typename Implementation>
template<typename ActionType>
Probability HypothesisStateInterface<Implementation>::get_probability(const HypothesisId& hypothesis,
                                                                      const AgentIdx& agent_idx,
                                                                      const ActionType& action) const {
 return StateInterface<Implementation>::impl().get_probability(hypothesis, agent_idx, action);
}

template<typename Implementation>
template<typename ActionType>
ActionType HypothesisStateInterface<Implementation>::get_last_action(const AgentIdx& agent_idx) const {
 return StateInterface<Implementation>::impl().get_last_action(agent_idx);
}

template<typename Implementation>
inline Probability HypothesisStateInterface<Implementation>::get_prior(const HypothesisId& hypothesis, const AgentIdx& agent_idx) const {
 return StateInterface<Implementation>::impl().get_prior(hypothesis, agent_idx);
}

template<typename Implementation>
inline HypothesisId HypothesisStateInterface<Implementation>::get_num_hypothesis(const AgentIdx& agent_idx) const {
 return StateInterface<Implementation>::impl().get_num_hypothesis(agent_idx);
}

template<typename Implementation>
inline HypothesisId HypothesisStateInterface<Implementation>::get_current_hypothesis(const AgentIdx& agent_idx) const {
 return current_agents_hypothesis_.at(agent_idx);
}


} // namespace mcts

#endif // MCTS_HYPOTHESIS_STATE_H