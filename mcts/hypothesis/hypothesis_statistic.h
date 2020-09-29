// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================

#ifndef MCTS_HYPOTHESIS_STATISTICS_H
#define MCTS_HYPOTHESIS_STATISTICS_H

#include <cmath>
#include <map>
#include <optional>

#include "mcts/mcts.h"
#include "mcts/hypothesis/common.h"
#include "mcts/hypothesis/hypothesis_state.h"
#include "mcts/mcts_parameters.h"

namespace mcts {

constexpr HypothesisId HYPOTHESIS_ID_NOT_SET = 100000;
class HypothesisStatistic : public mcts::NodeStatistic<HypothesisStatistic>,
                                   mcts::RandomGenerator,
                                   mcts::RequiresHypothesis
{
public:
    MCTS_TEST;

    HypothesisStatistic(ActionIdx num_actions, AgentIdx agent_idx, const MctsParameters& mcts_parameters) :
                    NodeStatistic<HypothesisStatistic>(num_actions, agent_idx, mcts_parameters),
                    RandomGenerator(mcts_parameters.RANDOM_SEED),
                    ego_cost_value_(0.0f),
                    latest_ego_cost_(0.0f),
                    ucb_statistics_(),
                    total_node_visits_hypothesis_(),
                    num_expanded_actions_(0),
                    total_node_visits_(0),
                    hypothesis_id_current_iteration_(HYPOTHESIS_ID_NOT_SET),
                    upper_cost_bound(mcts_parameters.hypothesis_statistic.UPPER_COST_BOUND),
                    lower_cost_bound(mcts_parameters.hypothesis_statistic.LOWER_COST_BOUND),
                    k_discount_factor(mcts_parameters.DISCOUNT_FACTOR), 
                    exploration_constant(mcts_parameters.hypothesis_statistic.EXPLORATION_CONSTANT),
                    cost_based_action_selection_(mcts_parameters.hypothesis_statistic.COST_BASED_ACTION_SELECTION),
                    progressive_widening_hypothesis_based_(mcts_parameters.hypothesis_statistic.PROGRESSIVE_WIDENING_HYPOTHESIS_BASED),
                    progressive_widening_k(mcts_parameters.hypothesis_statistic.PROGRESSIVE_WIDENING_K),
                    progressive_widening_alpha(mcts_parameters.hypothesis_statistic.PROGRESSIVE_WIDENING_ALPHA)
                    {}

    inline void init_hypothesis_variables(const HypothesisId hypothesis_id) {
        const auto it_count = total_node_visits_hypothesis_.find(hypothesis_id);
        if (it_count == total_node_visits_hypothesis_.end()) {
            total_node_visits_hypothesis_[hypothesis_id] = 0;
        }
        const auto it_pair = ucb_statistics_.find(hypothesis_id);
        if (it_pair == ucb_statistics_.end()) {
            auto& ucb_pair = ucb_statistics_[hypothesis_id]; // no init required
        }
    }

    template <class S>
    ActionIdx choose_next_action(const StateInterface<S>& state) {
        const HypothesisStateInterface<S>& impl = state.impl();
        hypothesis_id_current_iteration_ = impl.get_current_hypothesis(agent_idx_);

        if((progressive_widening_hypothesis_based_ &&
           require_progressive_widening_hypothesis_based(hypothesis_id_current_iteration_)) ||
           (!progressive_widening_hypothesis_based_ &&
           require_progressive_widening_total(hypothesis_id_current_iteration_))) {

            /* Init hypothesis node count and Q-Values if not visited under this hypothesis yet */
            init_hypothesis_variables(hypothesis_id_current_iteration_);

            /* Sample action from hypothesis */
            ActionIdx sampled_action = impl.plan_action_current_hypothesis(agent_idx_);

            /* Initialized UCBPair for this acion for this hypothesis (counts are updated during backprop.) */
            auto& ucb_pair = ucb_statistics_[hypothesis_id_current_iteration_][sampled_action];
            num_expanded_actions_ += 1;
            return sampled_action;
        } else {
            /* Select one action of previous actions */
            /*  Cost-based action selection out of hypothesis action set
            with highest ego cost to predict worst case behavior for this hypothesis (uses uct formula to also explore other actions) */
            if(cost_based_action_selection_) {
                /* Each hypothesis has worst case actions, hypothesis-specific worst case actions */
                if (progressive_widening_hypothesis_based_) {
                    return get_worst_case_action(ucb_statistics_.at(hypothesis_id_current_iteration_),
                            total_node_visits_hypothesis_.at(hypothesis_id_current_iteration_)).first;
                /* Worst case among all actions */
                } else {
                    return get_total_worst_case_action();
                }
            /* Random action-selection */    
            } else {
                /* Random sampling among hypothesis actions */
                if (progressive_widening_hypothesis_based_) {
                    return *sample_from_hypothesis_expanded_actions(hypothesis_id_current_iteration_);
                /* Random sampling among all actions */
                } else {
                    return sample_from_all_expanded_actions();
                }
            }
        }
    }

    void update_from_heuristic(const NodeStatistic<HypothesisStatistic>& heuristic_statistic)
    {
        const HypothesisStatistic& heuristic_statistic_impl = heuristic_statistic.impl();
        ego_cost_value_ = heuristic_statistic_impl.ego_cost_value_;
        latest_ego_cost_ = ego_cost_value_;
        auto& node_visits_hypothesis = total_node_visits_hypothesis_[hypothesis_id_current_iteration_];
        MCTS_EXPECT_TRUE(total_node_visits_ == 0); // This should be the first visit
        node_visits_hypothesis += 1;
        total_node_visits_ += 1;
    }

    void update_statistic(const NodeStatistic<HypothesisStatistic>& changed_child_statistic) {
        const HypothesisStatistic& changed_uct_statistic = changed_child_statistic.impl();

        //Action Value update step
        UcbPair& ucb_pair = ucb_statistics_[hypothesis_id_current_iteration_][collected_cost_.first]; // we remembered for which action we got the reward, must be the same as during backprop, if we linked parents and childs correctly
        //action value: Q'(s,a) = Q(s,a) + (latest_return - Q(s,a))/N =  1/(N+1 ( latest_return + N*Q(s,a))
        latest_ego_cost_ = collected_cost_.second + k_discount_factor * changed_uct_statistic.latest_ego_cost_;
        ucb_pair.action_count_ += 1;
        ucb_pair.action_ego_cost_ = ucb_pair.action_ego_cost_ + (latest_ego_cost_ - ucb_pair.action_ego_cost_) / ucb_pair.action_count_;
        VLOG_EVERY_N(6, 10) << "Agent "<< agent_idx_ <<", Action ego cost, action " << collected_cost_.first << ", C(s,a) = " << ucb_pair.action_ego_cost_;
        auto& node_visits_hypothesis = total_node_visits_hypothesis_[hypothesis_id_current_iteration_];
        node_visits_hypothesis += 1;
        ego_cost_value_ = ego_cost_value_ + (latest_ego_cost_ - ego_cost_value_) / node_visits_hypothesis;
        total_node_visits_ += 1;
    }

    ActionIdx get_best_action() const { throw std::logic_error("Not a meaningful call for this statistic");};

    Policy get_policy() const {
        Policy policy;
        for(const auto hyp_ub_stat : ucb_statistics_) {
            for (const auto& ucb_pair : hyp_ub_stat.second) {
                double action_cost_normalized = (ucb_pair.second.action_ego_cost_-lower_cost_bound)/(upper_cost_bound-lower_cost_bound); 
                policy[ucb_pair.first] = action_cost_normalized;
            }
        }
        return policy;
    }

    void set_heuristic_estimate(const Reward& accum_rewards, const Cost& accum_ego_cost) {
        ego_cost_value_ = accum_ego_cost;
    };

    std::string print_edge_information(const ActionIdx& action) const { return "";};

    std::string print_node_information() const
    {
        std::stringstream ss;
        ss << std::setprecision(2) << "V=" << ego_cost_value_ << ", N=" << total_node_visits_;
        return ss.str();
    }

    typedef struct UcbPair
    {
        explicit UcbPair() : action_count_(0), action_ego_cost_(0.0f) {};
        unsigned action_count_;
        double action_ego_cost_;
    } UcbPair;

    std::pair<ActionIdx, Cost> get_worst_case_action(const std::unordered_map<ActionIdx, UcbPair>& ucb_statistics, unsigned int node_visits) const
    {
        double largest_cost = std::numeric_limits<double>::min();
        ActionIdx worst_action = ucb_statistics.begin()->first;

        for (const auto& ucb_pair : ucb_statistics) 
        {
            double action_cost_normalized = (ucb_pair.second.action_ego_cost_-lower_cost_bound)/(upper_cost_bound-lower_cost_bound); 
            if(action_cost_normalized < 0 || action_cost_normalized > 1) {
              LOG(ERROR) << "Cost normalization wrong: " << action_cost_normalized << ", at ace=" << ucb_pair.second.action_ego_cost_ << ", lcb=" << 
                  lower_cost_bound << ", ucb=" << upper_cost_bound;
            }
            const double ucb_cost = action_cost_normalized + 2 * exploration_constant * sqrt( (2* std::log(node_visits)) / (ucb_pair.second.action_count_)  );
            if (ucb_cost > largest_cost) {
                largest_cost = ucb_cost;
                worst_action = ucb_pair.first;
            }
        }
        return std::make_pair(worst_action, largest_cost);
    }

    std::optional<ActionIdx> sample_from_hypothesis_expanded_actions(const HypothesisId& hypothesis) const {
        const auto& action_map = ucb_statistics_.at(hypothesis);
        if (action_map.empty()) {
            return std::nullopt;
        }
        std::uniform_int_distribution<ActionIdx> random_action_selection(0,action_map.size()-1);
        const auto& random_it = std::next(std::begin(action_map), random_action_selection(random_generator_));
        return std::optional<ActionIdx>{random_it->first};
    }

    ActionIdx sample_from_all_expanded_actions() const {
        for (const auto& hypothesis_uct : ucb_statistics_) {
            if(const auto hyp_sample = sample_from_hypothesis_expanded_actions(hypothesis_uct.first)) {
                return *hyp_sample;
            }
        }
        throw std::logic_error("No actions available to sample from.");
    }

    ActionIdx get_total_worst_case_action() const {
        double largest_cost = std::numeric_limits<double>::min();
        ActionIdx worst_action = ucb_statistics_.begin()->first;
        for (const auto& hypothesis_uct : ucb_statistics_) {
            if (hypothesis_uct.second.empty()) {
                continue;
            }
            const auto hypothesis_worst = get_worst_case_action(hypothesis_uct.second, total_node_visits_hypothesis_.at(hypothesis_uct.first));
            if (hypothesis_worst.second > largest_cost) {
                largest_cost = hypothesis_worst.second;
                worst_action = hypothesis_worst.first;
            }
        }
        return worst_action;
    }

    std::unordered_map<HypothesisId, std::unordered_map<ActionIdx, UcbPair>> get_ucb_statistics() const {
        return ucb_statistics_;
    }

    std::unordered_map<HypothesisId, unsigned int> get_total_node_visits() const {
        return total_node_visits_hypothesis_;
    }

private: // methods
    inline bool require_progressive_widening_hypothesis_based(const HypothesisId& hypothesis_id) const {
        const auto num_expanded = num_expanded_actions(hypothesis_id);
        const auto widening_term = progressive_widening_k * std::pow(num_node_visits(hypothesis_id),
                progressive_widening_alpha);
                // use progressive widening based on hypothesis-specific visit and action counts
        return num_expanded <= widening_term && num_expanded < num_actions_;
    }

    inline bool require_progressive_widening_total(const HypothesisId& hypothesis_id) const {
        const auto widening_term = progressive_widening_k * std::pow(total_node_visits_,
                progressive_widening_alpha);
                // At least one action should be expanded for each hypothesis,
                // otherwise use progressive widening based on total visit and action count
        return num_expanded_actions_ <= widening_term && num_expanded_actions_ < num_actions_;
    }

    // How many children exist based on specific hypothesis
    inline unsigned int num_expanded_actions(const HypothesisId& hypothesis_id) const {
        return ucb_statistics_.at(hypothesis_id).size();
    }

    // How often was this statistic visited under specific hypothesis
    inline unsigned int num_node_visits(const HypothesisId& hypothesis_id) const {
        return total_node_visits_hypothesis_.at(hypothesis_id);
    }
private: // members

    Reward ego_cost_value_; // average over all previous actions and heuristic calls going out from this node
    Reward latest_ego_cost_;   // tracks the ego cost during backpropagation (one action)
    std::unordered_map<HypothesisId, std::unordered_map<ActionIdx, UcbPair>> ucb_statistics_; // first: action selection count, action-ego_cost_qvalue
    std::unordered_map<HypothesisId, unsigned int> total_node_visits_hypothesis_;
    HypothesisId hypothesis_id_current_iteration_; // persist hypothesis id between action selection and backpropagation
    unsigned int total_node_visits_;
    unsigned int num_expanded_actions_;

    // PARAMS
    const Reward upper_cost_bound;
    const Reward lower_cost_bound;
    const double k_discount_factor;
    const double exploration_constant;

    const bool cost_based_action_selection_;
    const bool progressive_widening_hypothesis_based_;

    const double progressive_widening_k;
    const double progressive_widening_alpha;
};

} // namespace mcts


#endif // MCTS_HYPOTHESIS_STATISTICS_H