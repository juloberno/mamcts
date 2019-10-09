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
                     HypothesisStateInterface<HypothesisStatisticTestState>(current_agents_hypothesis) {}
    ~HypothesisStatisticTestState() {};


    HypothesisId get_num_hypothesis(const AgentIdx& agent_idx) const {return 2;}

    const std::vector<AgentIdx> get_agent_idx() const {
        return std::vector<AgentIdx>{0,1};
    }

    typedef int ActionType;

private:

};



#endif // BELIEF_TRACKER_TEST_STATE_H
