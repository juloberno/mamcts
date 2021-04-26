// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================

#ifndef BELIEF_TRACKER_TEST_STATE_H
#define BELIEF_TRACKER_TEST_STATE_H

#include <iostream>
#include "mcts/hypothesis/hypothesis_state.h"

using namespace mcts;

// A test state for belief tracker only with interfaces used by belief tracker
class HypothesisStatisticTestState : public mcts::HypothesisStateInterface<HypothesisStatisticTestState>
{
public:
    HypothesisStatisticTestState(const std::unordered_map<AgentIdx, HypothesisId>& current_agents_hypothesis) :
                     HypothesisStateInterface<HypothesisStatisticTestState>(current_agents_hypothesis),
                     use_first_action_(true) {}
    ~HypothesisStatisticTestState() {};

    ActionIdx plan_action_current_hypothesis(const AgentIdx& agent_idx) const {
        switch(current_agents_hypothesis_.at(agent_idx)) {
            case 0: 
                if(use_first_action_) {
                     return 5;
                }
                else {
                    return 3;
                }
                
            case 1: 
                if(use_first_action_) {
                     return 2;
                }
                else {
                    return 4;
                }
        }
    }

    double get_excution_step_length() const {
      return 1.0;
    }

    void choose_random_seed(const unsigned& seed_idx) {}

    void change_actions() {use_first_action_ = !use_first_action_;}

    const std::vector<AgentIdx> get_other_agent_idx() const {
        return std::vector<AgentIdx>{1};
    }

    const AgentIdx get_ego_agent_idx() const {
        return 0;
    }

    typedef int ActionType;

private:
    bool use_first_action_;

};



#endif // BELIEF_TRACKER_TEST_STATE_H
