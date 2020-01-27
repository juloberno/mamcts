// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================

#ifndef UCT_STATISTIC_H
#define UCT_STATISTIC_H

#include "mcts/mcts.h"
#include <iostream>
#include <iomanip>

namespace mcts {

// A upper confidence bound implementation
class UctStatistic : public mcts::NodeStatistic<UctStatistic>, mcts::RandomGenerator
{
public:
    MCTS_TEST

    UctStatistic(ActionIdx num_actions, AgentIdx agent_idx, const MctsParameters & mcts_parameters) :
             NodeStatistic<UctStatistic>(num_actions, agent_idx, mcts_parameters),
             RandomGenerator(mcts_parameters.RANDOM_SEED),
             value_(0.0f),
             latest_return_(0.0),
             ucb_statistics_([&]() -> std::map<ActionIdx, UcbPair>{
             std::map<ActionIdx, UcbPair> map;
             for (ActionIdx ai = 0; ai < num_actions; ++ai) { map[ai] = UcbPair();}
             return map;
             }()),
             total_node_visits_(0),
             unexpanded_actions_(num_actions),
             upper_bound(mcts_parameters.uct_statistic.UPPER_BOUND),
             lower_bound(mcts_parameters.uct_statistic.LOWER_BOUND),
             k_discount_factor(mcts_parameters.DISCOUNT_FACTOR), 
             k_exploration_constant(mcts_parameters.uct_statistic.EXPLORATION_CONSTANT) {
                 // initialize action indexes from 0 to (number of actions -1)
                 std::iota(unexpanded_actions_.begin(), unexpanded_actions_.end(), 0);
             }

    ~UctStatistic() {};

    template <class S>
    ActionIdx choose_next_action(const S& state) {
        if(unexpanded_actions_.empty())
        {
            // Select an action based on the UCB formula
            std::vector<double> values;
            calculate_ucb_values(ucb_statistics_, values);
            // find largest index
            ActionIdx selected_action = std::distance(values.begin(), std::max_element(values.begin(), values.end()));
            return selected_action;

        } else
        {
            // Select randomly an unexpanded action
            std::uniform_int_distribution<ActionIdx> random_action_selection(0,unexpanded_actions_.size()-1);
            ActionIdx array_idx = random_action_selection(random_generator_);
            ActionIdx selected_action = unexpanded_actions_[array_idx];
            unexpanded_actions_.erase(unexpanded_actions_.begin()+array_idx);
            return selected_action;
        }
    }

    ActionIdx get_best_action() {
        double temp = ucb_statistics_.begin()->second.action_value_;
        ActionIdx best = 0;
        for (auto it = ucb_statistics_.begin(); it != ucb_statistics_.end(); ++it)
        {
            if(it->second.action_value_>temp){
                temp = it->second.action_value_;
                best = it->first;
            }
        }
        return best;
    }

    void update_from_heuristic(const NodeStatistic<UctStatistic>& heuristic_statistic)
    {
        const UctStatistic& heuristic_statistic_impl = heuristic_statistic.impl();
        value_ = heuristic_statistic_impl.value_;
        latest_return_ = value_;
        MCTS_EXPECT_TRUE(total_node_visits_ == 0); // This should be the first visit
        total_node_visits_ += 1;
    }

    void update_statistic(const NodeStatistic<UctStatistic>& changed_child_statistic) {
        const UctStatistic& changed_uct_statistic = changed_child_statistic.impl();

        //Action Value update step
        UcbPair& ucb_pair = ucb_statistics_[collected_reward_.first]; // we remembered for which action we got the reward, must be the same as during backprop, if we linked parents and childs correctly
        //action value: Q'(s,a) = Q(s,a) + (latest_return - Q'(s,a))/N
        latest_return_ = collected_reward_.second + k_discount_factor * changed_uct_statistic.latest_return_;
        ucb_pair.action_count_ += 1;
        ucb_pair.action_value_ = ucb_pair.action_value_ + (latest_return_ - ucb_pair.action_value_) / ucb_pair.action_count_;
        
        total_node_visits_ += 1;
        value_ = value_ + (latest_return_ - value_) / total_node_visits_;
    }

    void set_heuristic_estimate(const Reward& accum_rewards, const Cost& accum_ego_cost)
    {
       value_ = accum_rewards;
    }

    std::string print_node_information() const
    {
        std::stringstream ss;
        ss << std::setprecision(2) << "V=" << value_ << ", N=" << total_node_visits_;
        return ss.str();
    }

    std::string print_edge_information(const ActionIdx& action ) const
    {
        std::stringstream ss;
        auto action_it = ucb_statistics_.find(action);
        if(action_it != ucb_statistics_.end()) {
            ss << std::setprecision(2) <<  "a=" << int(action) << ", N=" << action_it->second.action_count_ << ", V=" << action_it->second.action_value_;
        }
        return ss.str();
    }


    typedef struct UcbPair
    {
        UcbPair() : action_count_(0), action_value_(0.0f) {};
        unsigned action_count_;
        double action_value_;
    } UcbPair;

    void calculate_ucb_values(const std::map<ActionIdx, UcbPair>& ucb_statistics, std::vector<double>& values ) const
    {
        values.resize(ucb_statistics.size());

        for (size_t idx = 0; idx < ucb_statistics.size(); ++idx)
        {
            double action_value_normalized = (ucb_statistics.at(idx).action_value_-lower_bound)/(upper_bound-lower_bound); 
            MCTS_EXPECT_TRUE(action_value_normalized>=0);
            MCTS_EXPECT_TRUE(action_value_normalized<=1);
            values[idx] = action_value_normalized + 2 * k_exploration_constant * sqrt( (2* log(total_node_visits_)) / ( ucb_statistics.at(idx).action_count_)  );
        }
    }
private:

    double value_;
    double latest_return_;   // tracks the return during backpropagation
    std::map<ActionIdx, UcbPair> ucb_statistics_; // first: action selection count, action-value
    unsigned int total_node_visits_;
    std::vector<int> unexpanded_actions_; // contains all action indexes which have not been expanded yet

    // PARAMS
    const double upper_bound;
    const double lower_bound;
    const double k_discount_factor;
    const double k_exploration_constant;

};

} // namespace mcts

#endif