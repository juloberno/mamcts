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
class BeliefTrackerTestState : public mcts::HypothesisStateInterface<BeliefTrackerTestState>
{
public:
    BeliefTrackerTestState(const std::unordered_map<AgentIdx, HypothesisId>& current_agents_hypothesis) :
                     HypothesisStateInterface<BeliefTrackerTestState>(current_agents_hypothesis) {}
    ~BeliefTrackerTestState() {};

    template<typename ActionType = int>
    Probability get_probability(const HypothesisId& hypothesis, const AgentIdx& agent_idx, const ActionType& action) const {
        if(hypothesis == 0) {
            return get_prob_hy1(action);
        } else if(hypothesis==1) {
            return get_prob_hy2(action);
        } else {
            throw;
        }
    }

    template<typename ActionType = int>
    ActionType get_last_action(const AgentIdx& agent_idx) const {
        if(agent_idx == 0) {
            return get_last_action_ag1();
        } else if(agent_idx==1) {
            return get_last_action_ag2();
        } else {
            throw;
        }
    }

    Probability get_prior(const HypothesisId& hypothesis, const AgentIdx& agent_idx) const { 
        if(hypothesis == 0) {
            if(agent_idx == 0) {
                return get_prior_ag1_hy1();
            } else if(agent_idx==1) {
                return get_prior_ag2_hy1();
            } else {
                throw;
            }
        } else if(hypothesis==1) {
             if(agent_idx == 0) {
                return get_prior_ag1_hy2();
            } else if(agent_idx==1) {
                return get_prior_ag2_hy2();
            } else {
                throw;
            }
        } else {
            throw;
        }
    }

    HypothesisId get_num_hypothesis(const AgentIdx& agent_idx) const {return 2;}

    const std::vector<AgentIdx> get_agent_idx() const {
        return std::vector<AgentIdx>{0,1};
    }


private:
    template<typename ActionType = int>
    ActionType get_last_action_ag1() const {return 5;}

    template<typename ActionType = int>
    ActionType get_last_action_ag2() const {return 2;}

    Probability get_prior_ag1_hy1() const { return 0.5f;}

    Probability get_prior_ag1_hy2() const { return 0.7f;}

    Probability get_prior_ag2_hy1() const { return 0.6f;}

    Probability get_prior_ag2_hy2() const { return 0.4f;}

    template<typename ActionType = int>
    Probability get_prob_hy1(const ActionType& action) const { 
        switch(action) {
            case 5: return 0.3f;
            case 2: return 0.2f;      
        }
        return 0.0f;
    }

    template<typename ActionType = int>
    Probability get_prob_hy2(const ActionType& action) const { 
        switch(action) {
            case 5: return 0.7f;
            case 2: return 0.4f;      
        } 
        return 0.0f;
    }
};



#endif // BELIEF_TRACKER_TEST_STATE_H
