// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================

#ifndef SIMPLESTATE_H
#define SIMPLESTATE_H

#include <iostream>
#include "mcts/hypothesis/hypothesis_state.h"

using namespace mcts;

// A simple environment with a 1D state, only if both agents select different actions, they get nearer to the terminal state
class HypothesisSimpleState : public mcts::HypothesisStateInterface<HypothesisSimpleState>
{
public:
    HypothesisSimpleState(int length) : HypothesisStateInterface<HypothesisSimpleState>({}),
     state_length_(length), winning_state_length_(10), loosing_state_length_(-1) {};
    ~HypothesisSimpleState() {};

    std::shared_ptr<HypothesisSimpleState> clone() const
    {
        return std::make_shared<HypothesisSimpleState>(*this);
    }

    ActionIdx plan_action_current_hypothesis(const AgentIdx& agent_idx) const {
        return 0;
    };

    std::shared_ptr<HypothesisSimpleState> execute(const JointAction& joint_action, std::vector<Reward>& rewards ) const {
        // normally we map each single action value in joint action with a map to the floating point action. Here, not required
        rewards.resize(2);
        rewards[0] = 0; rewards[1] = 0;
        if(joint_action == JointAction{0,0} || joint_action == JointAction{1,1})
        {
            return std::make_shared<HypothesisSimpleState>(*this);
        }
        else if(joint_action == JointAction{0,1} || joint_action == JointAction{1,0})
        {

            //rewards[0] = -1.0f; rewards[1] = -1.0f;
            auto new_state_length = state_length_ + 1;

            rewards = std::vector<Reward>{1, 1};

            if(new_state_length >= winning_state_length_) {
                rewards = std::vector<Reward>{5, 10};
                new_state_length = winning_state_length_;
            }
            return std::make_shared<HypothesisSimpleState>(new_state_length);
        }
        else
        {
            std::cout << "unvalid action selected" << std::endl;
            return std::make_shared<HypothesisSimpleState>(*this);
        }

    }

    ActionIdx get_num_actions(AgentIdx agent_idx) const {
        return 2;
    }

    bool is_terminal() const {
        return state_length_ >= winning_state_length_ || state_length_ <= loosing_state_length_;
    }

    const std::vector<AgentIdx> get_agent_idx() const {
        return std::vector<AgentIdx>{0,1};
    }

    std::string sprintf() const
    {
        std::stringstream ss;
        ss << "SimpleState (state_length: " << state_length_ << ")";
        return ss.str();
    }
private:
    int state_length_;

    // PARAMS
    int winning_state_length_;
    int loosing_state_length_;
};



#endif 
