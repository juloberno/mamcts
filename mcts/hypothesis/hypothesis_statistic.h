// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================

#ifndef MCTS_HYPOTHESIS_STATISTICS_H
#define MCTS_HYPOTHESIS_STATISTICS_H

#include <cmath>
#include <map>

#include "mcts/mcts.h"
#include "mcts/hypothesis/common.h"
#include "mcts/hypothesis/hypothesis_state.h"

namespace mcts {

constexpr HypothesisId HYPOTHESIS_ID_NOT_SET = 100000;
constexpr double EGO_COST_VALUE_NOT_SET = 100000;
constexpr double LATEST_EGO_COST_NOT_SET = 100000;

class HypothesisStatistic : public mcts::NodeStatistic<HypothesisStatistic>,
                                   mcts::RandomGenerator,
                                   mcts::RequiresHypothesis
{
public:
    MCTS_TEST;

    HypothesisStatistic(ActionIdx num_actions, AgentIdx agent_idx) :
                    NodeStatistic<HypothesisStatistic>(num_actions, agent_idx),
                    ego_cost_value_(EGO_COST_VALUE_NOT_SET),
                    latest_ego_cost_(LATEST_EGO_COST_NOT_SET),
                    ucb_statistics_(),
                    total_node_visits_(),
                    hypothesis_id_current_iteration_(HYPOTHESIS_ID_NOT_SET),
                    upper_cost_bound(mcts::MctsParameters::HypothesisStatistic::UPPER_COST_BOUND),
                    lower_cost_bound(mcts::MctsParameters::HypothesisStatistic::LOWER_COST_BOUND),
                    k_discount_factor(mcts::MctsParameters::DISCOUNT_FACTOR), 
                    k_exploration_constant(mcts::MctsParameters::HypothesisStatistic::EXPLORATION_CONSTANT),
                    cost_based_action_selection_(mcts::MctsParameters::HypothesisStatistic::COST_BASED_ACTION_SELECTION),
                    progressive_widening_k(mcts::MctsParameters::HypothesisStatistic::PROGRESSIVE_WIDENING_K),
                    progressive_widening_alpha(mcts::MctsParameters::HypothesisStatistic::PROGRESSIVE_WIDENING_ALPHA)
                    {}

    inline void init_hypothesis_variables(const HypothesisId hypothesis_id) {
        const auto it_count = total_node_visits_.find(hypothesis_id);
        if (it_count != total_node_visits_.end()) {
            total_node_visits_[hypothesis_id] = 0;
        }
        const auto it_pair = ucb_statistics_.find(hypothesis_id);
        if (it_pair != ucb_statistics_.end()) {
            auto& ucb_pair = ucb_statistics_[hypothesis_id]; // no init required
        }
    }

    template <class S>
    ActionIdx choose_next_action(const StateInterface<S>& state, std::vector<int>& unexpanded_actions) {
        const HypothesisStateInterface<S>& impl = state.impl();
        hypothesis_id_current_iteration_ = impl.get_current_hypothesis(agent_idx_);
        /* Init hypothesis node count and Q-Values if not visited under this hypothesis yet */
        init_hypothesis_variables(hypothesis_id_current_iteration_);

        if(require_progressive_widening(hypothesis_id_current_iteration_)) {
            /* Sample new action:
            1) Sample action from hypothesis
            2) Initialized UCBPair for this acion for this hypothesis (counts are updated during backprop.)
            3) Return this action */
            ActionIdx sampled_action = impl.plan_action_current_hypothesis(agent_idx_);
            auto& ucb_pair = ucb_statistics_[hypothesis_id_current_iteration_][sampled_action];
            return sampled_action;
        } else {
            /* Select an action based on previous actions */
            if(cost_based_action_selection_) {
                /*  Cost-based action selection out of hypothesis action set
                with highest ego cost to predict worst case behavior for this hypothesis (uses uct formula to also explore other actions) */
                return get_worst_case_action(uct_statistics_[hypothesis_id_current_iteration_], total_node_visits_[hypothesis_id_current_iteration_];)
            } else {
           /* Random action-selection just use one action  */    

           }
        }
        
    }

    void update_from_heuristic(const NodeStatistic<UctStatistic>& heuristic_statistic)
    {
        const HypothesisStatistic& heuristic_statistic_impl = heuristic_statistic.impl();
        ego_cost_value_ = heuristic_statistic_impl.ego_cost_value_;
        latest_ego_cost_ = ego_cost_value_;
        auto& node_visits_hypothesis = total_node_visits_[hypothesis_id_current_iteration_];
        MCTS_EXPECT_TRUE(total_node_visits_ == 0); // This should be the first visit
        node_visits_hypothesis += 1;
    }

    void update_statistic(const NodeStatistic<UctStatistic>& changed_child_statistic) {
        const HypothesisStatistic& changed_uct_statistic = changed_child_statistic.impl();

        //Action Value update step
        UcbPair& ucb_pair = ucb_statistics_[hypothesis_id_current_iteration_][collected_cost_.first]; // we remembered for which action we got the reward, must be the same as during backprop, if we linked parents and childs correctly
        //action value: Q'(s,a) = Q(s,a) + (latest_return - Q'(s,a))/N
        latest_ego_cost_ = collected_cost_.second + k_discount_factor * changed_uct_statistic.latest_ego_cost_;
        ucb_pair.action_count_ += 1;
        ucb_pair.action_ego_cost_ = ucb_pair.action_ego_cost_ + (latest_ego_cost_ - ucb_pair.action_ego_cost_) / ucb_pair.action_count_;
        
        auto& node_visits_hypothesis = total_node_visits_[hypothesis_id_current_iteration_];
        node_visits_hypothesis += 1;
        ego_cost_value_ = ego_cost_value_ + (latest_ego_cost_ - ego_cost_value_) / node_visits_hypothesis;
    }

    
    ActionIdx get_best_action() { throw std::logic_error("Not at meaningful call for this statistic");};

    void set_heuristic_estimate(const Reward& accum_rewards, const Cost& accum_ego_cost) {
        ego_cost_value_ = accum_ego_cost;
    };

    std::string print_node_information() const {return "";};

    std::string print_edge_information(const ActionIdx& action) const { return "";};

    typedef struct UcbPair
    {
        UcbPair() : action_count_(0), action_ego_cost_(0.0f) {};
        unsigned action_count_;
        double action_ego_cost_;
    } UcbPair;

    ActionIdx get_worst_case_action(const std::map<ActionIdx, UcbPair>& ucb_statistics, unsigned int node_visits) const
    {
        double largest_cost = std::numeric_limits<double>::min();
        ActionIdx worst_action = ucb_statistics.begin()->first;

        for (const auto& ucb_pair : ucb_statistics) 
        {
            double action_cost_normalized = (ucb_pair.second.action_ego_cost_-lower_cost_bound)/(upper_cost_bound-lower_cost_bound); 
            MCTS_EXPECT_TRUE(action_value_normalized>=0);
            MCTS_EXPECT_TRUE(action_value_normalized<=1);
            const double ucb_cost = action_cost_normalized + 2 * k_exploration_constant * sqrt( (2* std::log(total_node_visits_)) / (ucb_pair.second.action_count_)  );
            if (ucb_cost > largest_cost) {
                largest_cost = ucb_cost;
                worst_action = ucb_pair.first;
            }
        }
        return worst_action;
    }

private: // methods
    inline bool require_progressive_widening(const HypothesisId& hypothesis_id) const {
        return num_expanded_actions(hypothesis_id) <= 
                progressive_widening_k * std::pow(num_node_visits(hypothesis_id),
                progressive_widening_alpha);
    }

    // How many children exist based on specific hypothesis
    inline unsigned int num_expanded_actions(const HypothesisId& hypothesis_id) const {
        return ucb_statistics_.at(hypothesis_id).size();
    }

    // How often was this statistic visited under specific hypothesis
    inline unsigned int num_node_visits(const HypothesisId& hypothesis_id) const {
        return total_node_visits_.at(hypothesis_id);
    }
private: // members

    double ego_cost_value_; // average over all previous actions and heuristic calls going out from this node
    double latest_ego_cost_;   // tracks the ego cost during backpropagation (one action)
    std::unordered_map<HypothesisId, std::unordered_map<ActionIdx, UcbPair>> ucb_statistics_; // first: action selection count, action-ego_cost_qvalue
    std::unordered_map<HypothesisId, unsigned int> total_node_visits_;
    HypothesisId hypothesis_id_current_iteration_; // persist hypothesis id between action selection and backpropagation

    // PARAMS
    const double upper_cost_bound;
    const double lower_cost_bound;
    const double k_discount_factor;
    const double k_exploration_constant;

    const bool cost_based_action_selection_;

    const double progressive_widening_k;
    const double progressive_widening_alpha;
};

} // namespace mcts


#endif // MCTS_HYPOTHESIS_STATISTICS_H