// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================

#ifndef UCT_STATISTIC_H
#define UCT_STATISTIC_H

#include "mcts/mcts.h"
#include <unordered_map>
#include <iostream>
#include <iomanip>

namespace mcts {

// A upper confidence bound implementation
class UctStatistic : public mcts::NodeStatistic<UctStatistic>, mcts::RandomGenerator
{
public:
    MCTS_TEST
    #ifdef UCT_COST_CONSTRAINED_STATISTIC_H
    friend class CostConstrainedStatistic;
    #endif

    typedef struct UcbPair
    {
        UcbPair() : action_count_(0), action_value_(0.0f), init_value_(0.0) {};
        UcbPair(unsigned count, double value, double init_value) : 
            action_count_(count), action_value_(value), init_value_(init_value) {};
        UcbPair(unsigned count, double value) : 
            action_count_(count), action_value_(value), init_value_(value) {};
        unsigned action_count_;
        double action_value_;
        double init_value_; // rembembers first value, e.g. from a heuristic initialization
    } UcbPair;
    typedef std::map<ActionIdx, UcbPair> UcbStatistics;

    UctStatistic(ActionIdx num_actions, AgentIdx agent_idx, const MctsParameters & mcts_parameters) :
             NodeStatistic<UctStatistic>(num_actions, agent_idx, mcts_parameters),
             RandomGenerator(mcts_parameters.RANDOM_SEED),
             value_(0.0f),
             latest_return_(0.0),
             ucb_statistics_(),
             total_node_visits_(0),
             unexpanded_actions_(num_actions),
             upper_bound(mcts_parameters.uct_statistic.UPPER_BOUND),
             lower_bound(mcts_parameters.uct_statistic.LOWER_BOUND),
             k_discount_factor(mcts_parameters.DISCOUNT_FACTOR), 
             exploration_constant(mcts_parameters.uct_statistic.EXPLORATION_CONSTANT),
             progressive_widening_k(mcts_parameters.uct_statistic.PROGRESSIVE_WIDENING_K),
             progressive_widening_alpha(mcts_parameters.uct_statistic.PROGRESSIVE_WIDENING_ALPHA),
             use_bound_estimation_(mcts_parameters.USE_BOUND_ESTIMATION) {
                 // initialize action indexes from 0 to (number of actions -1)
                 std::iota(unexpanded_actions_.begin(), unexpanded_actions_.end(), 0);
             }

    ~UctStatistic() {};

    template <class S>
    ActionIdx choose_next_action(const S& state) {
        if(require_progressive_widening_total()) {
            // Select randomly an unexpanded action
            std::uniform_int_distribution<ActionIdx> random_action_selection(0,unexpanded_actions_.size()-1);
            ActionIdx array_idx = random_action_selection(random_generator_);
            ActionIdx selected_action = unexpanded_actions_[array_idx];
            unexpanded_actions_.erase(unexpanded_actions_.begin()+array_idx);
            ucb_statistics_[selected_action] = UcbPair();
            return selected_action;
        } else {
            // Select an action based on the UCB formula
            std::unordered_map<ActionIdx, double> values;
            ActionIdx selected_action = calculate_ucb_and_max_action(ucb_statistics_, values);
            return selected_action;
       }
    }

    ActionIdx get_best_action() const {
        double temp = ucb_statistics_.begin()->second.action_value_;
        ActionIdx best = ucb_statistics_.begin()->first;
        for (auto it = ucb_statistics_.begin(); it != ucb_statistics_.end(); ++it)
        {
            if(it->second.action_value_>temp){
                temp = it->second.action_value_;
                best = it->first;
            }
        }
        return best;
    }

    Policy get_policy() const {
        Policy policy;
        for (auto it = ucb_statistics_.begin(); it != ucb_statistics_.end(); ++it)
        {
            policy[it->first] = it->second.action_value_;
        }
        return policy;
    }

    void update_from_heuristic(const NodeStatistic<UctStatistic>& heuristic_statistic) {
        const UctStatistic& heuristic_statistic_impl = heuristic_statistic.impl();
        update_from_heuristic_from_backpropagated(heuristic_statistic_impl.value_);
    }


    void update_from_heuristic_from_backpropagated(const Reward& backpropagated_value) {
        value_ = backpropagated_value;
        latest_return_ = value_;
        total_node_visits_ += 1;
    }

    void update_from_heuristic_from_backpropagated(const Reward& backpropagated_value, const UcbStatistics& ucb_statistics) {
        update_from_heuristic_from_backpropagated(backpropagated_value);
        ucb_statistics_ = ucb_statistics;
    }

    void update_statistic(const NodeStatistic<UctStatistic>& changed_child_statistic) {
        const UctStatistic& changed_uct_statistic = changed_child_statistic.impl();
        this->update_statistics_from_backpropagated(changed_uct_statistic.latest_return_);
    }

    std::pair<Reward, Reward> calculate_bounds_based_on_stat(const UcbStatistics& ucb_stats) const {
        const auto minmax_return = std::minmax_element(ucb_stats.begin(), ucb_stats.end(), [](const auto& ucb1, const auto& ucb2) {
              return ucb1.second.action_value_ < ucb2.second.action_value_;});
        std::pair<Reward, Reward> bounds{minmax_return.first->second.action_value_, 
                                         minmax_return.second->second.action_value_};
        if (bounds.first == bounds.second) {
            if(bounds.second != 0.0) return std::make_pair(Reward(0.0), Reward(bounds.second));
            else return std::make_pair(Reward(0.0), Reward(1.0));   
        }
        return bounds;
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

    void set_heuristic_estimate(const Reward& accum_rewards, const EgoCosts& accum_ego_cost)
    {
      this->set_heuristic_estimate_from_backpropagated(accum_rewards);
    }


    void set_heuristic_estimate_from_backpropagated(const Reward& value) {
       value_ = value;
    }

    void set_heuristic_estimate_from_backpropagated(const std::unordered_map<ActionIdx, Reward>& action_returns) {
        double val = 0.0;
        for(const auto action_value : action_returns) {
            val += action_value.second;
            ucb_statistics_[action_value.first] = UcbPair(1, action_value.second, action_value.second);
        }
        value_ = val / action_returns.size();
    }

    void merge_node_statistics(const std::vector<UctStatistic>& statistics) {
        ucb_statistics_.clear();
        // Sum over all actions for statistics
        for (const auto& stat : statistics) {
            for (const auto& action_value : stat.get_ucb_statistics()) {
                ucb_statistics_[action_value.first].action_value_ += action_value.second.action_value_;
                ucb_statistics_[action_value.first].action_count_ += 1;
            }
        }
        // Average action values and node value
        value_ = 0.0;
        for (auto& action_value : ucb_statistics_) {
                action_value.second.action_value_ /= action_value.second.action_count_;
                value_ += action_value.second.action_value_;
        }
        value_ /= ucb_statistics_.size();
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

    Reward get_normalized_ucb_value(const ActionIdx& action) const {
      
        std::pair<Reward, Reward> bounds(lower_bound, upper_bound);
        if(use_bound_estimation_) {
            bounds = calculate_bounds_based_on_stat(ucb_statistics_);
        }
        double action_value_normalized =  (ucb_statistics_.at(action).action_value_-bounds.first)/(bounds.second-bounds.first); 
        
        LOG_IF(FATAL, action_value_normalized>1 || action_value_normalized < 0) << "Wrong action value normalization: " << action_value_normalized;
        return action_value_normalized;
    }

    Reward get_reward_lower_bound() const {
      return lower_bound;
    }

    Reward get_reward_upper_bound() const {
      return upper_bound;
    }

    ActionIdx calculate_ucb_and_max_action(const UcbStatistics& ucb_statistics, std::unordered_map<ActionIdx, double>& values) const {
        values.reserve(ucb_statistics.size());
        ActionIdx maximizing_action = 0;
        double max_value = std::numeric_limits<double>::min();

        std::pair<Reward, Reward> bounds(lower_bound, upper_bound);
        if(use_bound_estimation_) {
            bounds = calculate_bounds_based_on_stat(ucb_statistics);
        }

        for (const auto ucb_pair : ucb_statistics) {   
            double action_value_normalized = (ucb_pair.second.action_value_-bounds.first)/(bounds.second-bounds.first); 
            MCTS_EXPECT_TRUE(action_value_normalized>=0);
            MCTS_EXPECT_TRUE(action_value_normalized<=1);
            values[ucb_pair.first] = action_value_normalized + 2 * exploration_constant * sqrt( (2* log(total_node_visits_)) / ( ucb_pair.second.action_count_)  );
            if (values[ucb_pair.first] > max_value) {
                max_value = values[ucb_pair.first];
                maximizing_action = ucb_pair.first;
            }
        }
        return maximizing_action;
    }

    std::string sprintf() const {
        std::stringstream ss;
        for(const auto& ucb_stat : ucb_statistics_) {
            ss << "a=" <<  ucb_stat.first << ", q=" << ucb_stat.second.action_value_ << ", n=" << ucb_stat.second.action_count_ << "|";
        }
        return ss.str();
    }

    const UcbStatistics& get_ucb_statistics() const { return ucb_statistics_; }

     inline bool require_progressive_widening_total() const {
        const auto widening_term = progressive_widening_k * std::pow(total_node_visits_,
                progressive_widening_alpha);
                // At least one action should be expanded for each hypothesis,
                // otherwise use progressive widening based on total visit and action count
        return num_expanded_actions() <= widening_term && num_expanded_actions() < num_actions_;
    }

    // How many children exist based on specific hypothesis
    inline unsigned int num_expanded_actions() const {
        return ucb_statistics_.size();
    }

    void SetUcbStatistics(const UcbStatistics& ucb_stats) { ucb_statistics_ = ucb_stats; }


protected:
    double value_;
    double latest_return_;   // tracks the return during backpropagation
    
    UcbStatistics ucb_statistics_; // first: action selection count, action-value
    unsigned int total_node_visits_;
    std::vector<ActionIdx> unexpanded_actions_; // contains all action indexes which have not been expanded yet

    // PARAMS
    double upper_bound;
    double lower_bound;
    double k_discount_factor;
    double exploration_constant;

    double progressive_widening_k;
    double progressive_widening_alpha;

    bool use_bound_estimation_;

};

} // namespace mcts


#endif