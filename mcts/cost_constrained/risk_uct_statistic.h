// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================

#ifndef RISK_UCT_STATISTIC_H
#define RISK_UCT_STATISTIC_H

#include "mcts/statistics/uct_statistic.h"
#include <iostream>
#include <iomanip>

namespace mcts {

// A upper confidence bound implementation
class RiskUctStatistic : public UctStatistic
{
public:
    MCTS_TEST
    FRIEND_COST_CONSTRAINED_STATISTIC

    RiskUctStatistic(ActionIdx num_actions, AgentIdx agent_idx, const MctsParameters & mcts_parameters) :
             UctStatistic(num_actions, agent_idx, mcts_parameters) { }

    ~RiskUctStatistic() {};

    void update_from_heuristic_from_backpropagated(const Reward& backpropagated) {
        value_ = backpropagated;
        latest_return_ = value_;
        total_node_visits_ += 1;
    }
    
    void update_statistics_from_backpropagated(const Reward& backpropagated) {
        //Action risk backpropagation requires different handling than standard UCT backpropagation
        UcbPair& ucb_pair = ucb_statistics_[collected_reward_.first]; // we remembered for which action we got the reward, must be the same as during backprop, if we linked parents and childs correctly
        // action risk: roh'(s,a) = 1/transition_count_a' * ( roh(s')  + transition_count_a*roh(s,a))
        // risk of next state roh(s') := collected_risk + backpropagated (either heuristic found a risk OR during expansion to terminal state)
        // transition count a in current iteration: transition_count_a'
        // transition count a in previos iteration: transition_count_a
        const auto& transition_count_a_previous = collected_action_transition_counts_.first;
        const auto& transition_count_a_current = collected_action_transition_counts_.second;
        ucb_pair.action_count_ += 1;
        auto additional_risk = backpropagated;
        if (transition_count_a_previous != transition_count_a_current) {
            // a new sample was drawn for this action at this state -> consider resulting action cost
            additional_risk += collected_reward_.second;
        }
        const auto previous_action_value = ucb_pair.action_value_;
        ucb_pair.action_value_ = 1.0 / double(transition_count_a_current) * (additional_risk + transition_count_a_previous * ucb_pair.action_value_);

        if ( ucb_pair.action_value_ > 1.0) {
            bool test = true;
        }

        // remove previous risk of state s used in state before s and replace by new risk 
        double previous_backpropagated_risk;
        if(transition_count_a_previous == 0) {
          previous_backpropagated_risk = value_;  // This is the first transition before only heuristic was backpropagated 
        } else {
          previous_backpropagated_risk = previous_action_value;
        }
        const auto new_backpropagated_risk = ucb_pair.action_value_;
        latest_return_ = -1.0 * previous_backpropagated_risk + new_backpropagated_risk;
        LOG(INFO) << "pa=" << previous_action_value << ", av=" << ucb_pair.action_value_ << ", ptc=" << transition_count_a_previous << ", ctc=" << transition_count_a_current <<
                ", v=" << value_ << ", ar=" << additional_risk;
        total_node_visits_ += 1;
    }

    void set_heuristic_estimate_from_backpropagated(const Reward& backpropagated) {
       value_ = backpropagated;
    }
};

} // namespace mcts


#endif // RISK_UCT_STATISTIC_H