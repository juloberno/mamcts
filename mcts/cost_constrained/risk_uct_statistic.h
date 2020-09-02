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
        //Action Value update step
        UcbPair& ucb_pair = ucb_statistics_[collected_reward_.first]; // we remembered for which action we got the reward, must be the same as during backprop, if we linked parents and childs correctly
        //action value: Q'(s,a) = Q(s,a) + (latest_return - Q(s,a))/N =  1/(N+1 ( latest_return + N*Q(s,a))
        latest_return_ = collected_reward_.second + k_discount_factor * backpropagated;
        ucb_pair.action_count_ += 1;
        ucb_pair.action_value_ = ucb_pair.action_value_ + (latest_return_ - ucb_pair.action_value_) / ucb_pair.action_count_;
        VLOG_EVERY_N(6, 10) << "Agent "<< agent_idx_ <<", Action reward, action " << collected_cost_.first << ", Q(s,a) = " << ucb_pair.action_value_;
        total_node_visits_ += 1;
        value_ = value_ + (latest_return_ - value_) / total_node_visits_;
    }

    void set_heuristic_estimate_from_backpropagated(const Reward& backpropagated) {
       value_ = backpropagated;
    }
};

} // namespace mcts


#endif // RISK_UCT_STATISTIC_H