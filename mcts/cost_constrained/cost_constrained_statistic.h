// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================

#ifndef UCT_COST_CONSTRAINED_STATISTIC_H
#define UCT_COST_CONSTRAINED_STATISTIC_H

#include "mcts/statistics/uct_statistic.h"
#include "mcts/cost_constrained/risk_uct_statistic.h"
#include "mcts/cost_constrained/lp_single_cost_solver.h"
#include "mcts/cost_constrained/lp_multi_cost_solver.h"
#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>

namespace mcts {

#define CONSTRAINT_COST_IDX 0 // 1D linear program only over this cost idx calculated

inline PolicySampled sample_policy(const Policy& policy,
           std::mt19937& random_generator) {
      std::vector<int> discrete_probability_weights;
      std::vector<ActionIdx> action_order;
      for (const auto actions : policy) {
          discrete_probability_weights.push_back(std::nearbyint(
                      actions.second*1000.0));
          action_order.push_back(actions.first);
      }

      std::discrete_distribution<> action_dist(discrete_probability_weights.begin(),
                                          discrete_probability_weights.end());
      const auto sampled_action = action_order.at(action_dist(random_generator));
      return std::make_pair(sampled_action, policy);
    }

class CostConstrainedStatistic : public mcts::NodeStatistic<CostConstrainedStatistic>, mcts::RandomGenerator
{
public:
    MCTS_TEST

    CostConstrainedStatistic(ActionIdx num_actions, AgentIdx agent_idx, const MctsParameters & mcts_parameters) :
             NodeStatistic<CostConstrainedStatistic>(num_actions, agent_idx, mcts_parameters),
             RandomGenerator(mcts_parameters.RANDOM_SEED),
             reward_statistic_(num_actions, agent_idx,  make_reward_statistic_parameters(mcts_parameters)),
             cost_statistics_(),
             unexpanded_actions_(),
             mean_step_costs_(),
             exploration_policy_(),
             kappa(mcts_parameters.cost_constrained_statistic.KAPPA),
             action_filter_factor(mcts_parameters.cost_constrained_statistic.ACTION_FILTER_FACTOR),
             cost_constraints_(mcts_parameters.cost_constrained_statistic.COST_CONSTRAINTS),
             exploration_reduction_factor_(mcts_parameters.cost_constrained_statistic.EXPLORATION_REDUCTION_FACTOR),
             exploration_constant_offset_(mcts_parameters.cost_constrained_statistic.EXPLORATION_REDUCTION_CONSTANT_OFFSET),
             exploration_reduction_init_(mcts_parameters.cost_constrained_statistic.EXPLORATION_REDUCTION_INIT),
             min_visits_policy_ready_(mcts_parameters.cost_constrained_statistic.MIN_VISITS_POLICY_READY),
             use_cost_thresholding_(mcts_parameters.cost_constrained_statistic.USE_COST_THRESHOLDING),
             use_chance_constrained_updates_(mcts_parameters.cost_constrained_statistic.USE_CHANCE_CONSTRAINED_UPDATES),
             cost_tresholds_(mcts_parameters.cost_constrained_statistic.COST_THRESHOLDS),
             use_lambda_policy_(mcts_parameters.cost_constrained_statistic.USE_LAMBDA_POLICY)
             {
               // initialize action indexes from 0 to (number of actions -1)
                for(auto action_idx = 0; action_idx < num_actions; ++action_idx) {
                  unexpanded_actions_.push_back(action_idx);
                }
                for(const auto action : unexpanded_actions_) {
                  exploration_policy_[action] = 1/unexpanded_actions_.size(); // default uniform exploration across actions
                }
              }

    ~CostConstrainedStatistic() {};

    template <class S>
    ActionIdx choose_next_action(const S& state) {
       set_step_length(state.get_execution_step_length());
       if( policy_is_ready())
        {
          // Expansion policy does consider node counts
          return greedy_policy(kappa, action_filter_factor, false).first;
        } else {
            // Select randomly an unexpanded action
            std::uniform_int_distribution<ActionIdx> random_action_selection(0,unexpanded_actions_.size()-1);
            ActionIdx array_idx = random_action_selection(random_generator_);
            ActionIdx selected_action = unexpanded_actions_[array_idx];
            unexpanded_actions_.erase(unexpanded_actions_.begin()+array_idx);
            return selected_action;
        }
    }

    Policy get_policy() const {
      return greedy_policy(0.0f, action_filter_factor, false).second;
    }

    ActionIdx get_best_action() const {
      return greedy_policy(0.0f, action_filter_factor, false).first;
    }

    bool policy_is_ready() const {
      if (min_visits_policy_ready_ > 0 && reward_statistic_.total_node_visits_ > min_visits_policy_ready_) {
        return true;
      } else {
        return unexpanded_actions_.empty();
      }
    }

    void set_step_length(const double& step_length) {
      step_length_ = step_length;
    }

    PolicySampled greedy_policy(const double kappa_local, const double action_filter_factor_local, bool combined_must_be_in_search_policy) const {
      const auto allowed_actions = cost_thresholding_action_selection();

      if(!use_lambda_policy_) {
        const auto action = calculate_ucb_maximizing_action(allowed_actions, kappa_local);
        return PolicySampled(action, Policy({{action, 1.0}}));
      }
  
      std::unordered_map<ActionIdx, double> ucb_values_with_exploration;
      std::unordered_map<ActionIdx, double> ucb_values_without_exploration;
      calculate_ucb_values_with_lambda(ucb_values_with_exploration, ucb_values_without_exploration,
                                     kappa_local, allowed_actions);
      
      const auto feasible_actions = filter_feasible_actions(ucb_values_with_exploration,
                                                            ucb_values_without_exploration,
                                                             action_filter_factor_local);
      auto policy = solve_LP_and_sample(feasible_actions, combined_must_be_in_search_policy);
      return policy;
    }

    Cost calc_updated_constraint_based_on_policy(const PolicySampled& policy, const Cost& current_constraint, const Cost& mean_step_cost) const {
      double other_actions_costs = 0.0f;
      for(const auto& action_pair : policy.second) {
        if(action_pair.first == policy.first) {
          continue;
        }
        other_actions_costs += action_pair.second * cost_statistics_.at(CONSTRAINT_COST_IDX).ucb_statistics_.at(action_pair.first).action_value_;
      }
      return (current_constraint - policy.second.at(policy.first)* - other_actions_costs) /
              (cost_statistics_.at(CONSTRAINT_COST_IDX).k_discount_factor * policy.second.at(policy.first));
    }

    std::vector<Cost> calc_updated_constraints_based_on_policy(const PolicySampled& policy, const std::vector<Cost>& current_constraints) const {
      std::vector<Cost> new_constraints;
      for (std::size_t cost_idx = 0; cost_idx < current_constraints.size(); ++cost_idx) {
        new_constraints.push_back(calc_updated_constraint_based_on_policy(policy, current_constraints.at(cost_idx),
             mean_step_costs_.at(policy.first).at(cost_idx)));
      }
      return new_constraints;
    }

    ActionIdx calculate_ucb_maximizing_action(const std::vector<ActionIdx>& allowed_actions, const double& kappa_local) const {
      ActionIdx maximizing_action = allowed_actions.at(0);
      Reward maximizing_value = std::numeric_limits<Reward>::lowest();
      for (const auto& action_idx  : allowed_actions) {
          double reward_value_normalized = reward_statistic_.get_normalized_ucb_value(action_idx);

          const auto exploration_term = kappa_local * 
              sqrt( log(reward_statistic_.total_node_visits_) / ( reward_statistic_.ucb_statistics_.at(action_idx).action_count_));
          const auto value = reward_value_normalized + (std::isnan(exploration_term) ? std::numeric_limits<double>::max() : exploration_term);
          if(value > maximizing_value) {
            maximizing_action = action_idx;
            maximizing_value = value;
          }
      }
      return maximizing_action;
    }

    void calculate_ucb_values_with_lambda(std::unordered_map<ActionIdx, double>& values_with_exploration,
                            std::unordered_map<ActionIdx, double>& values_without_exploration,
                            const double& kappa_local,
                            std::vector<ActionIdx> allowed_actions = std::vector<ActionIdx>()) const {
      const auto& reward_stats = reward_statistic_.ucb_statistics_;
      const auto& cost_stats = cost_statistics_.at(0).ucb_statistics_;
      MCTS_EXPECT_TRUE(reward_stats.size() ==  cost_stats.size());

      values_with_exploration.reserve(reward_statistic_.ucb_statistics_.size());
      values_without_exploration.reserve(reward_statistic_.ucb_statistics_.size());
      if(allowed_actions.empty()) {
        allowed_actions.resize(reward_statistic_.ucb_statistics_.size());
        std::transform(reward_statistic_.ucb_statistics_.begin(), reward_statistic_.ucb_statistics_.end(), 
                   allowed_actions.begin(),[](auto p) {return p.first; });
      }

      for (const auto& action_idx  : allowed_actions) {
          double reward_value_normalized = reward_statistic_.get_normalized_ucb_value(action_idx);
          
          const auto exploration_term = kappa_local * 
              sqrt( log(reward_statistic_.total_node_visits_) / ( reward_statistic_.ucb_statistics_.at(action_idx).action_count_));
          
          double cost_lambda_term = 0.0;
          for (std::size_t cost_idx = 0; cost_idx < cost_statistics_.size(); ++cost_idx) {
            cost_lambda_term += cost_statistics_.at(cost_idx).get_ucb_statistics().at(action_idx).action_value_ * 
                  mcts_parameters_.cost_constrained_statistic.LAMBDAS.at(cost_idx);
          }
          values_without_exploration[action_idx] = reward_value_normalized - cost_lambda_term;
          values_with_exploration[action_idx] = values_without_exploration[action_idx]
                                         + (std::isnan(exploration_term) ? std::numeric_limits<double>::max() : exploration_term);
      }
    }

    Policy consider_exploration_policy(const Policy& search_policy, bool must_be_in_search_policy) const {
      const auto node_visits = cost_statistics_.at(0).total_node_visits_;
      const auto mix_prob = get_exploration_mixture_probability(node_visits);
      Policy combined_policy;
      for(auto& action : exploration_policy_) {
        if (search_policy.find(action.first) != search_policy.end()) {
          combined_policy[action.first] = (1-mix_prob)*search_policy.at(action.first) + mix_prob * exploration_policy_.at(action.first);
        } else if(!must_be_in_search_policy) {
          combined_policy[action.first] = mix_prob * exploration_policy_.at(action.first);
        }
      }
      return combined_policy;
    }

    PolicySampled solve_LP_and_sample(const std::vector<ActionIdx>& feasible_actions, bool combined_must_be_in_search_policy) const {
      Policy search_policy;
      if (feasible_actions.size() == 1) {
          Policy policy;
          const auto& cost_stats = cost_statistics_.at(0).get_ucb_statistics();
          for ( const auto action : cost_stats) {
              policy[action.first] = 0.0;
          }
          policy[feasible_actions.at(0)] = 1.0;
          search_policy = policy;
      } else if (cost_statistics_.size() > 1) {
          search_policy = lp_multiple_cost_solver(feasible_actions, cost_statistics_, cost_constraints_,
           mcts_parameters_.cost_constrained_statistic.LAMBDAS, random_generator_,
              std::max(*std::max_element(cost_constraints_.begin(), cost_constraints_.end()), 1.0));
          
      } else {
              lp_single_cost_solver(feasible_actions, cost_statistics_.at(CONSTRAINT_COST_IDX),
          cost_constraints_.at(CONSTRAINT_COST_IDX), random_generator_);
      }
      const auto combined_policy = consider_exploration_policy(search_policy, combined_must_be_in_search_policy);
      return sample_policy(combined_policy, random_generator_);
    }

    std::vector<ActionIdx> filter_feasible_actions(const std::unordered_map<ActionIdx, double>& values_with_exploration,
                                          const std::unordered_map<ActionIdx, double>& values_without_exploration,
                                           const double& action_filter_factor_local) const {
      std::vector<ActionIdx> filtered_actions;
      const auto maximizing_action_it = std::max_element(values_with_exploration.begin(), values_with_exploration.end(), [](const auto& p1, const auto& p2){
                 return p1.second < p2.second; });
      const ActionIdx maximizing_action = maximizing_action_it->first;
      const Reward max_val = values_without_exploration.at(maximizing_action);
      const double node_counts_maximizing = (reward_statistic_.ucb_statistics_.at(maximizing_action).action_count_ == 0) ? std::numeric_limits<double>::max() :
                                 sqrt( log( reward_statistic_.ucb_statistics_.at(maximizing_action).action_count_) /
                                                  ( reward_statistic_.ucb_statistics_.at(maximizing_action).action_count_) );
      for (const auto action_value : values_without_exploration) {
          const double value_difference = std::abs(action_value.second - max_val);
          const double node_count_relations = ( reward_statistic_.ucb_statistics_.at(action_value.first).action_count_ == 0) ? std::numeric_limits<double>::max() :
                                 sqrt( log( reward_statistic_.ucb_statistics_.at(action_value.first).action_count_) /
                                                  ( reward_statistic_.ucb_statistics_.at(action_value.first).action_count_) ) + 
                                              node_counts_maximizing;
          if(value_difference <= action_filter_factor_local * node_count_relations) {
            filtered_actions.push_back(action_value.first);
          }
      }
      return filtered_actions;
    }

    

    static double calculate_next_lambda(const double& current_lambda,
                                        const double& gradient_update_step,
                                        const double& cost_constraint,
                                        const double& tau_gradient_clip,
                                        const double& discount_factor,
                                        const double& cost_sampled_action,
                                        const double& reward_lower_bound,
                                        const double& reward_upper_bound) {
        const double gradient = (cost_sampled_action - cost_constraint);
        const double new_lambda = current_lambda + gradient_update_step * gradient;
        const double clip_upper_limit = (reward_upper_bound - reward_lower_bound) /
                                         (tau_gradient_clip * ( 1 - discount_factor));
        const double clipped_new_lambda = std::min(std::max(new_lambda, double(0.0f)), clip_upper_limit);
        VLOG_EVERY_N(5, 10) << "Norm. UCBSampled: " << cost_sampled_action << ", grad = "
                             << gradient << ", step = " << gradient_update_step <<", clipp_upper_lim = " << 
                            clip_upper_limit << ", clipped_lambda = " << clipped_new_lambda;
        return clipped_new_lambda;
    }

    std::vector<ActionIdx> cost_thresholding_action_selection() const {
      std::vector<ActionIdx> allowed_actions;
      allowed_actions.resize(cost_statistics_.at(0).ucb_statistics_.size());
      std::transform(cost_statistics_.at(0).ucb_statistics_.begin(), cost_statistics_.at(0).ucb_statistics_.end(), 
                  allowed_actions.begin(),[](auto p) {return p.first; });
      if(use_cost_thresholding_.empty()) {
        return allowed_actions;
      }
      for(std::size_t cost_stat_idx = 0; cost_stat_idx < cost_statistics_.size(); ++cost_stat_idx) {
        if(!use_cost_thresholding_.at(cost_stat_idx)) {
          continue;
        }
        // Also track current lowest action to return it if allowed actions are empty
        Cost current_lowest_value = cost_statistics_.at(cost_stat_idx).ucb_statistics_.begin()->second.action_value_;
        ActionIdx current_lowest_action = cost_statistics_.at(cost_stat_idx).ucb_statistics_.begin()->first;
        for(const auto cost_ucb : cost_statistics_.at(cost_stat_idx).ucb_statistics_) {
          const auto comparison_uct_value = cost_statistics_.at(cost_stat_idx).get_normalized_ucb_value(cost_ucb.first);
          if(comparison_uct_value < current_lowest_value) {
            current_lowest_action = cost_ucb.first;
          }
          if(comparison_uct_value > cost_tresholds_.at(cost_stat_idx)) {
            allowed_actions.erase(std::remove(allowed_actions.begin(), allowed_actions.end(), cost_ucb.first), allowed_actions.end());
            if(allowed_actions.empty()) {
              return std::vector<ActionIdx>(1, current_lowest_action);
            }
          }
        }
      }
      return allowed_actions;
    }

    void init_cost_statistics(std::size_t size) {
      if(cost_statistics_.empty()) {
        for(std::size_t idx = 0; idx < size; ++idx) {
          cost_statistics_.push_back(RiskUctStatistic(num_actions_, agent_idx_, make_cost_statistics_parameters(mcts_parameters_)));
        }
        ActionIdx action_idx = 0;
        while(action_idx < num_actions_) {
          mean_step_costs_[action_idx].resize(size, 0.0);
          action_idx++;
        }
      }
    }

    void update_from_heuristic(const NodeStatistic<CostConstrainedStatistic>& heuristic_statistic)
    {
      const CostConstrainedStatistic& statistic_impl = heuristic_statistic.impl();

      const auto& heuristic_reward_value = statistic_impl.reward_statistic_.value_;
      const auto& heuristic_ucb_stats = statistic_impl.reward_statistic_.ucb_statistics_;
      reward_statistic_.update_from_heuristic_from_backpropagated(heuristic_reward_value, heuristic_ucb_stats);

      const auto& cost_stats_heuristic = statistic_impl.cost_statistics_;
      init_cost_statistics(cost_stats_heuristic.size());
      for (auto cost_stat_idx = 0; cost_stat_idx < cost_stats_heuristic.size(); ++cost_stat_idx) {
        const auto& heuristic_cost_value = cost_stats_heuristic.at(cost_stat_idx).value_;
        const auto& backpropagated_step_length = cost_stats_heuristic.at(cost_stat_idx).backpropagated_step_length_;
        const auto& heuristic_ucb_stats = cost_stats_heuristic.at(cost_stat_idx).ucb_statistics_;
        cost_statistics_[cost_stat_idx].update_from_heuristic_from_backpropagated(heuristic_cost_value, backpropagated_step_length,
                                                                                  heuristic_ucb_stats);
      }
    }

    void update_statistic(const NodeStatistic<CostConstrainedStatistic>& changed_child_statistic) {
      const CostConstrainedStatistic& statistic_impl = changed_child_statistic.impl();

      const auto reward_latest_return = statistic_impl.reward_statistic_.latest_return_;
      reward_statistic_.collected_reward_ = collected_reward_;
      reward_statistic_.update_statistics_from_backpropagated(reward_latest_return);

      const auto& cost_stats_child = statistic_impl.cost_statistics_;
      init_cost_statistics(cost_stats_child.size());
      for (auto cost_stat_idx = 0; cost_stat_idx < cost_stats_child.size(); ++cost_stat_idx) {
        const auto cost_latest_return = statistic_impl.cost_statistics_.at(cost_stat_idx).latest_return_;
        const auto backpropagated_step_length = statistic_impl.cost_statistics_.at(cost_stat_idx).backpropagated_step_length_;
        cost_statistics_[cost_stat_idx].collected_reward_ = std::pair<ActionIdx, Cost>(collected_cost_.first,
                                                                      collected_cost_.second[cost_stat_idx]);
        cost_statistics_[cost_stat_idx].collected_action_transition_counts_ = collected_action_transition_counts_;

        bool chance_update = !use_chance_constrained_updates_.empty() && use_chance_constrained_updates_.at(cost_stat_idx);
        cost_statistics_[cost_stat_idx].set_step_length(step_length_);
        cost_statistics_[cost_stat_idx].update_statistics_from_backpropagated(cost_latest_return, chance_update, backpropagated_step_length);

        mean_step_costs_[collected_cost_.first][cost_stat_idx] += (collected_cost_.second[cost_stat_idx] - mean_step_costs_[collected_cost_.first][cost_stat_idx]) /
                                                    (cost_statistics_.at(cost_stat_idx).ucb_statistics_[collected_cost_.first].action_count_);
      }
    }

    void set_heuristic_estimate(const Reward& accum_rewards, const EgoCosts& accum_ego_cost, double backpropated_step_length)
    {
      reward_statistic_.set_heuristic_estimate_from_backpropagated(accum_rewards);
    
      init_cost_statistics(accum_ego_cost.size());
      for (auto cost_stat_idx = 0; cost_stat_idx < accum_ego_cost.size(); ++cost_stat_idx) {
        cost_statistics_.at(cost_stat_idx).set_heuristic_estimate_from_backpropagated(
            accum_ego_cost.at(cost_stat_idx), backpropated_step_length);

      }
    }

    void set_heuristic_estimate(const std::unordered_map<ActionIdx, Reward> &action_returns,
                                const std::unordered_map<ActionIdx, EgoCosts>& action_costs,
                                const std::unordered_map<ActionIdx, double>& backpropated_step_length)
    {
      reward_statistic_.set_heuristic_estimate_from_backpropagated(action_returns);
    
      init_cost_statistics(action_costs.begin()->second.size());
      for (auto cost_stat_idx = 0; cost_stat_idx < action_costs.begin()->second.size(); ++cost_stat_idx) {
        std::unordered_map<ActionIdx, Reward> action_costs_stat;
        for (const auto& action_value : action_costs) {
          action_costs_stat[action_value.first] = action_value.second.at(cost_stat_idx);
        }
        cost_statistics_.at(cost_stat_idx).set_heuristic_estimate_from_backpropagated(action_costs_stat, backpropated_step_length);
      }
    }

    std::string print_node_information() const {
        return "";
    }

    static std::string print_policy(const Policy& policy) {
      std::stringstream ss;
      ss << "Policy: ";
      for (const auto& action_pair : policy) {
        ss << "P(a=" << action_pair.first << ") = " << action_pair.second << ", ";
      }
      return ss.str();
    }

    std::vector<mcts::Cost> expected_policy_cost(const Policy& policy) const {
      std::vector<mcts::Cost> expected_cost(2, 0.0);
      for ( std::size_t cost_idx; cost_idx < cost_statistics_.size(); ++cost_idx) {
        const auto& cost_stats = cost_statistics_.at(cost_idx);
        const auto& ucb_stats = cost_stats.ucb_statistics_;
        for(const auto& ucb_stat : ucb_stats) {
          expected_cost[cost_idx] += policy.at(ucb_stat.first) * ucb_stat.second.action_value_;
        } 
      }
      return expected_cost;
    }

    std::string print_edge_information(const ActionIdx& action) const {
        const auto& reward_stats = reward_statistic_.ucb_statistics_;
        std::unordered_map<ActionIdx, double> ucb_values_with_exploration;
        std::unordered_map<ActionIdx, double> ucb_values_without_exploration;
        calculate_ucb_values_with_lambda(ucb_values_with_exploration, 
                          ucb_values_without_exploration, 0.0f);
        std::stringstream ss;
        ss  << "Reward stats: " << reward_statistic_.sprintf() << "\n"
            << "Cost stats: ";
            for(std::size_t cost_stat_idx = 0; cost_stat_idx < cost_statistics_.size(); ++cost_stat_idx) {
              ss << cost_stat_idx <<  ") [" << cost_statistics_.at(cost_stat_idx).sprintf() << "]  ";
            } 
            ss << "\n" << "Lambdas:" << mcts_parameters_.cost_constrained_statistic.LAMBDAS << "\n"
            << "Ucb values: " << ucb_values_with_exploration << "\n"
            << "Mean step cost: C(a=" << action << ") = " << mean_step_costs_.at(action) << "\n";
        return ss.str();
    }

    Reward get_normalized_cost_action_value(const ActionIdx& action) const {
      return cost_statistics_.at(CONSTRAINT_COST_IDX).get_normalized_ucb_value(action);
    }

    MctsParameters make_cost_statistics_parameters(const MctsParameters& mcts_parameters) const {
      MctsParameters cost_statistics_parameters(mcts_parameters);
      cost_statistics_parameters.uct_statistic.LOWER_BOUND = mcts_parameters.cost_constrained_statistic.COST_LOWER_BOUND;
      cost_statistics_parameters.uct_statistic.UPPER_BOUND = mcts_parameters.cost_constrained_statistic.COST_UPPER_BOUND;

      cost_statistics_parameters.DISCOUNT_FACTOR = 1.0; //< For risk estimation, we do not apply a discount
      return cost_statistics_parameters;
    }

    MctsParameters make_reward_statistic_parameters(const MctsParameters& mcts_parameters) const {
      MctsParameters reward_statistic_parameters(mcts_parameters);
      reward_statistic_parameters.uct_statistic.LOWER_BOUND = mcts_parameters.cost_constrained_statistic.REWARD_LOWER_BOUND;
      reward_statistic_parameters.uct_statistic.UPPER_BOUND = mcts_parameters.cost_constrained_statistic.REWARD_UPPER_BOUND;
      return reward_statistic_parameters;
    }

    const UctStatistic::UcbStatistics& get_cost_ucb_statistics(const unsigned int& cost_index) const {
      return cost_statistics_.at(cost_index).ucb_statistics_;
    }

    unsigned get_num_costs() const { return cost_statistics_.size(); }

    const UctStatistic::UcbStatistics& get_reward_ucb_statistics() const {
      return reward_statistic_.ucb_statistics_;
    }

    const UctStatistic& get_reward_statistic() const {
      return reward_statistic_;
    }

    const UctStatistic& get_cost_statistic(const unsigned int& cost_index) const {
      return cost_statistics_.at(cost_index);
    }

    std::string sprintf() const {
      std::stringstream ss;
      ss << "Cost statistic: ";
      for(const auto& cost_statistic : cost_statistics_) {
        ss << cost_statistic.sprintf() << ", ";
      }
      return ss.str();
    }

    
    void merge_node_statistics(const std::vector<CostConstrainedStatistic>& statistics) {
      reward_statistic_.merge_node_statistics([&](){
        std::vector<UctStatistic> uct_statistics;
        for ( const auto& cost_stat : statistics) {
          uct_statistics.push_back(cost_stat.get_reward_statistic());
        }
        return uct_statistics;
      }());
      cost_statistics_.clear();
      init_cost_statistics(statistics.begin()->get_num_costs());
      for (unsigned cost_idx = 0; cost_idx < cost_statistics_.size(); ++cost_idx) {
        UctStatistic& this_uct = cost_statistics_[cost_idx];
        this_uct.merge_node_statistics([&](){
          std::vector<UctStatistic> uct_statistics;
          for ( const auto& cost_stat : statistics) {
            const UctStatistic& uct_other = cost_stat.get_cost_statistic(cost_idx);
            uct_statistics.push_back(uct_other);
          }
          return uct_statistics;
        }());
      }
    }

    void set_exploration_policy(const Policy& policy) {
      exploration_policy_ = policy;
    }

    Policy get_exploration_policy() const {
      return exploration_policy_;
    }


    double get_action_filter_factor() const { return action_filter_factor; }
    double get_kappa() const { return kappa; }
    double get_exploration_mixture_probability(const unsigned& total_node_visits) const {
      return std::max(std::min(exploration_reduction_init_ - 
            exploration_reduction_factor_*(total_node_visits - exploration_constant_offset_), 1.0), 0.0);

    }


private:

    UctStatistic reward_statistic_;
    std::vector<RiskUctStatistic> cost_statistics_;
    std::vector<ActionIdx> unexpanded_actions_;
    std::unordered_map<ActionIdx, std::vector<Cost>> mean_step_costs_;
    Policy exploration_policy_;
    double step_length_;

    const double kappa;
    const double action_filter_factor;
    const std::vector<double> cost_constraints_;
    const double exploration_reduction_factor_;
    const double exploration_constant_offset_;
    const double exploration_reduction_init_;
    const unsigned min_visits_policy_ready_;

    const std::vector<bool> use_cost_thresholding_;
    const std::vector<bool> use_chance_constrained_updates_;
    const std::vector<double> cost_tresholds_;
    const bool use_lambda_policy_;
};

template <>
inline void NodeStatistic<CostConstrainedStatistic>::update_statistic_parameters(MctsParameters& parameters,
                                            const CostConstrainedStatistic& root_statistic,
                                            const unsigned int& current_iteration) {
  if(!root_statistic.policy_is_ready()) {
    return;
  }
  const double gradient_update_step = parameters.cost_constrained_statistic.GRADIENT_UPDATE_STEP/(current_iteration + 1);
  const auto cost_constraints = parameters.cost_constrained_statistic.COST_CONSTRAINTS;
  const double tau_gradient_clip = parameters.cost_constrained_statistic.TAU_GRADIENT_CLIP;
  MCTS_EXPECT_TRUE(cost_constraints.size() == parameters.cost_constrained_statistic.LAMBDAS.size());
  for (std::size_t cost_idx = 0; cost_idx < parameters.cost_constrained_statistic.LAMBDAS.size(); ++cost_idx) {
    const double current_lambda = parameters.cost_constrained_statistic.LAMBDAS.at(cost_idx);
    const ActionIdx policy_sampled_action = root_statistic.greedy_policy(0.0f, 0.0f, true).first;
    const double normalized_cost_sampled_action = root_statistic.get_cost_statistic(cost_idx).
                    get_ucb_statistics().at(policy_sampled_action).action_value_;
    const double new_lambda =  CostConstrainedStatistic::calculate_next_lambda(current_lambda,
                                                                            gradient_update_step,
                                                                            cost_constraints.at(cost_idx),
                                                                            tau_gradient_clip,
                                                                            parameters.DISCOUNT_FACTOR,
                                                                            normalized_cost_sampled_action,
                                                                            root_statistic.get_reward_statistic().get_reward_lower_bound(),
                                                                            root_statistic.get_reward_statistic().get_reward_upper_bound());
    parameters.cost_constrained_statistic.LAMBDAS[cost_idx] = new_lambda;
    VLOG_EVERY_N(5, 100) << "Updated lambda" << cost_idx << "from " << current_lambda << 
      " to " << new_lambda << " in iteration " << current_iteration;
  }
}

template <>
inline MctsParameters NodeStatistic<CostConstrainedStatistic>::merge_mcts_parameters(std::vector<MctsParameters> parameters) {
  MctsParameters merged_parameters = *parameters.begin(); 
  // only lambda must be merged (being the only parameter changed iteratively)
  const auto num_lambdas = merged_parameters.cost_constrained_statistic.LAMBDAS.size();
  std::vector<double> merged_lambdas(num_lambdas, 0.0);
  for(std::size_t cost_idx = 0; cost_idx < num_lambdas; ++cost_idx) {
    for (const auto& mcts_params : parameters) {
      merged_lambdas[cost_idx] += mcts_params.cost_constrained_statistic.LAMBDAS[cost_idx];
    }
    merged_lambdas[cost_idx] /= parameters.size();
  }
  
  merged_parameters.cost_constrained_statistic.LAMBDAS = merged_lambdas;
  return merged_parameters;
}

} // namespace mcts

#endif