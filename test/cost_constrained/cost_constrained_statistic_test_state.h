// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================

#ifndef COST_CONSTRAINED_TEST_STATE_H
#define COST_CONSTRAINED_TEST_STATE_H

#include <iostream>
#include "mcts/state.h"

using namespace mcts;

// A simple environment with a 1D state, only if both agents select different actions, they get nearer to the terminal state
class CostConstrainedStatisticTestState : public mcts::StateInterface<CostConstrainedStatisticTestState>
{
public:
    CostConstrainedStatisticTestState(int length) : state_length_(length), winning_state_length_(10), loosing_state_length_(-1) {};
    ~CostConstrainedStatisticTestState() {};

    std::shared_ptr<CostConstrainedStatisticTestState> clone() const
    {
        return std::make_shared<CostConstrainedStatisticTestState>(*this);
    }

    std::shared_ptr<CostConstrainedStatisticTestState> execute(const JointAction& joint_action, std::vector<Reward>& rewards, Cost& ego_cost) const {
        // normally we map each single action value in joint action with a map to the floating point action. Here, not required
        rewards.resize(2);
        rewards[0] = 0; rewards[1] = 0;
        if(joint_action == JointAction{0,0} || joint_action == JointAction{1,1})
        {
            return std::make_shared<CostConstrainedStatisticTestState>(*this);
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
            return std::make_shared<CostConstrainedStatisticTestState>(new_state_length);
        }
        else
        {
            std::cout << "unvalid action selected" << std::endl;
            return std::make_shared<CostConstrainedStatisticTestState>(*this);
        }

    }

    ActionIdx get_num_actions(AgentIdx agent_idx) const {
        return 2;
    }

    bool is_terminal() const {
        return state_length_ >= winning_state_length_ || state_length_ <= loosing_state_length_;
    }

    const std::vector<AgentIdx> get_other_agent_idx() const {
        return std::vector<AgentIdx>{5};
    }

    const AgentIdx get_ego_agent_idx() const {
        return 4;
    }


    std::string sprintf() const
    {
        std::stringstream ss;
        ss << "CostConstrainedStatisticTestState (state_length: " << state_length_ << ")";
        return ss.str();
    }
private:
    int state_length_;

    // PARAMS
    int winning_state_length_;
    int loosing_state_length_;
};



#endif 
