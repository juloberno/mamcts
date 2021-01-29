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
             UctStatistic(num_actions, agent_idx, mcts_parameters),
             step_length_(),
             backpropagated_step_length_(0.0) { }

    ~RiskUctStatistic() {};

    void update_from_heuristic_from_backpropagated(const Reward& backpropagated, double backpropagated_step_length,
                                                  const UcbStatistics& ucb_statistics) {
        value_ = backpropagated;
        latest_return_ = value_;
        backpropagated_step_length_ = backpropagated_step_length;
        ucb_statistics_ = ucb_statistics;
        total_node_visits_ += 1;
    }
    
    void update_statistics_from_backpropagated(const Reward& backpropagated, bool chance_update, double backpropagated_step_length) {
        //Action Value update step
        UcbPair& ucb_pair = ucb_statistics_[collected_reward_.first]; // we remembered for which action we got the reward, must be the same as during backprop, if we linked parents and childs correctly
        //action value: Q'(s,a) = Q(s,a) + (latest_return - Q(s,a))/N =  1/(N+1 ( latest_return + N*Q(s,a))
        ucb_pair.action_count_ += 1;
 
        if (chance_update) {
            latest_return_ = std::max(collected_reward_.second, backpropagated);
        } else {
            latest_return_ = collected_reward_.second + backpropagated;
        }
        backpropagated_step_length_ = backpropagated_step_length + step_length_;
        auto step_length_normalized_return = latest_return_ / backpropagated_step_length_;
        ucb_pair.action_value_ = ucb_pair.action_value_ + (step_length_normalized_return - ucb_pair.action_value_) / ucb_pair.action_count_;
        VLOG_EVERY_N(6, 10) << "Agent "<< agent_idx_ <<", Action reward, action " << collected_cost_.first << ", Q(s,a) = " << ucb_pair.action_value_
            << ", normalized_return=" << step_length_normalized_return << ", back_prop_l" << backpropagated_step_length_ << ", step length =" << step_length_;
        
        total_node_visits_ += 1;
        value_ = value_ + (latest_return_ - value_) / total_node_visits_;
    }

    void set_heuristic_estimate_from_backpropagated(const Reward& value, double backpropagated_step_length) {
        value_ = value;
        backpropagated_step_length_ = backpropagated_step_length;
    }

    void set_heuristic_estimate_from_backpropagated(const std::unordered_map<ActionIdx, Reward>& action_returns,
                                                   const std::unordered_map<ActionIdx, double>& action_step_lengths) {
        double val = 0.0;
        double step = 0.0;
        for(const auto action_value : action_returns) {
            // Initialize statistics with step normalized returns, 
            // if step normalized returns are empty, we assume that action returns are already risk-based
            auto step_normalized_return = action_step_lengths.empty() ? action_value.second :
                                               action_value.second / action_step_lengths.at(action_value.first);
            ucb_statistics_[action_value.first] = UcbPair(1, step_normalized_return, step_normalized_return);
            val += action_value.second;
            step += action_step_lengths.at(action_value.first);
        }
        value_ = val / action_returns.size();
        backpropagated_step_length_ = step / action_returns.size();
    }

    void set_step_length(double step_length) {
        step_length_ = step_length;
    }

    private:
        double step_length_;
        double backpropagated_step_length_;
};

} // namespace mcts


#endif // RISK_UCT_STATISTIC_H