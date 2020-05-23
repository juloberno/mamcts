// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================

#ifndef UCT_COST_CONSTRAINED_STATISTIC_H
#define UCT_COST_CONSTRAINED_STATISTIC_H

#include "mcts/statistics/uct_statistic.h"
#include <iostream>
#include <iomanip>
#include <random>

namespace mcts {

// A upper confidence bound implementation
class CostConstrainedStatistic : public mcts::NodeStatistic<CostConstrainedStatistic>, mcts::RandomGenerator
{
public:
    MCTS_TEST
    FRIEND_COST_CONSTRAINED_STATISTIC

    CostConstrainedStatistic(ActionIdx num_actions, AgentIdx agent_idx, const MctsParameters & mcts_parameters) :
             NodeStatistic<CostConstrainedStatistic>(num_actions, agent_idx, mcts_parameters),
             reward_statistic_(num_actions, agent_idx,  make_reward_statistic_parameters(mcts_parameters)),
             cost_statistic_(num_actions, agent_idx, make_cost_statistic_parameters(mcts_parameters)),
             RandomGenerator(mcts_parameters.RANDOM_SEED),
             lambda(mcts_parameters.cost_constrained_statistic.LAMBDA),
             exploration_constant(mcts_parameters.cost_constrained_statistic.EXPLORATION_CONSTANT),
             action_filter_factor(mcts_parameters.cost_constrained_statistic.ACTION_FILTER_FACTOR),
             cost_constraint(mcts_parameters.cost_constrained_statistic.COST_CONSTRAINT)
             {
             }

    ~CostConstrainedStatistic() {};

    template <class S>
    ActionIdx choose_next_action(const S& state) {
      return get_best_action();
    }

    ActionIdx get_best_action() const {
      // Greedy Policy
      std::vector<double> ucb_values;
      calculate_ucb_values(ucb_values);
      const auto feasible_actions = filter_feasible_actions(ucb_values);
      const ActionIdx sampled_action = solve_LP_and_sample(feasible_actions);
      return sampled_action;
    }

    void calculate_ucb_values(std::vector<double>& values) const {
      const auto& reward_stats = reward_statistic_.ucb_statistics_;
      const auto& cost_stats = cost_statistic_.ucb_statistics_;
      MCTS_EXPECT_TRUE(reward_stats.size() ==  cost_stats.size());

      values.resize(reward_statistic_.ucb_statistics_.size(), 0.0f);
      for (size_t idx = 0; idx < reward_stats.size(); ++idx)
      {
          double cost_value_normalized = cost_statistic_.get_normalized_ucb_value(idx);
          double reward_value_normalized = reward_statistic_.get_normalized_ucb_value(idx);

          values[idx] = reward_value_normalized - lambda * cost_value_normalized 
                 + 2 * exploration_constant * sqrt( log(reward_statistic_.total_node_visits_) / ( reward_statistic_.ucb_statistics_.at(idx).action_count_)  );
      }
    }

    std::vector<ActionIdx> filter_feasible_actions(const std::vector<double>& values) const {
      std::vector<ActionIdx> filtered_actions;
      const ActionIdx maximizing_action = std::distance(values.begin(), std::max_element(values.begin(), values.end()));
      const Reward max_val = values[maximizing_action];
      const double node_counts_maximizing = sqrt( log( reward_statistic_.ucb_statistics_.at(maximizing_action).action_count_) /
                                                  ( reward_statistic_.ucb_statistics_.at(maximizing_action).action_count_) );
      for (size_t action_idx = 0; action_idx < values.size(); ++action_idx) {
          const double value_difference = values[action_idx] - max_val;
          const double node_count_relations = sqrt( log( reward_statistic_.ucb_statistics_.at(action_idx).action_count_) /
                                                  ( reward_statistic_.ucb_statistics_.at(action_idx).action_count_) ) + 
                                              node_counts_maximizing;
          if(value_difference <= action_filter_factor * node_count_relations) {
            filtered_actions.push_back(action_idx);
          }
      }
      return filtered_actions;
    }

    ActionIdx solve_LP_and_sample(const std::vector<ActionIdx>& feasible_actions) const {
      // Solved for K=1
      const auto& cost_stats = cost_statistic_.ucb_statistics_;
      auto stat_compare = [](const std::pair<const long unsigned int, mcts::UctStatistic::UcbPair>& a,
                             const std::pair<const long unsigned int, mcts::UctStatistic::UcbPair>& b) {
                                return a.second.action_value_ < b.second.action_value_; };
      const ActionIdx maximizing_action = std::distance(cost_stats.begin(), std::max_element(cost_stats.begin(), cost_stats.end(), stat_compare));
      const ActionIdx minimizing_action = std::distance(cost_stats.begin(), std::min_element(cost_stats.begin(), cost_stats.end(), stat_compare));

      // Three cases
      const double max_val = cost_stats.at(maximizing_action).action_value_;
      const double min_val = cost_stats.at(minimizing_action).action_value_;
      if( min_val >= cost_constraint) {
          // amin gets probability one, amax gets probability zero
          return minimizing_action;
      } else if( max_val <= cost_constraint) {
         // amax gets probability one, amin gets probability zero
         return maximizing_action;
      } else {
         const double probability_maximizer = (cost_constraint - min_val) / (max_val - min_val);
         std::uniform_real_distribution<double> unif(0, 1);
         double sample = unif(random_generator_);
         if(sample <= probability_maximizer) {
           return maximizing_action;
         } else {
           return minimizing_action;
         }
      }
    }

    static double calculate_next_lambda(const double& current_lambda,
                                        const double& gradient_update_step,
                                        const double& cost_constraint,
                                        const double& tau_gradient_clip,
                                        const CostConstrainedStatistic& root_statistic) {
        const ActionIdx policy_sampled_action = root_statistic.get_best_action();
        const double new_lambda = current_lambda + gradient_update_step * (
             root_statistic.get_normalized_cost_action_value(policy_sampled_action) - cost_constraint);
        const double clip_upper_limit = (root_statistic.reward_statistic_.upper_bound - root_statistic.reward_statistic_.lower_bound) /
                                         (tau_gradient_clip * ( 1 - root_statistic.reward_statistic_.k_discount_factor));
        const double clipped_new_lambda = std::min(std::max(new_lambda, double(0.0f)), clip_upper_limit);
        return clipped_new_lambda;
    }

    void update_from_heuristic(const NodeStatistic<CostConstrainedStatistic>& heuristic_statistic)
    {
      const CostConstrainedStatistic& statistic_impl = heuristic_statistic.impl();

      const auto heuristic_reward_value = statistic_impl.reward_statistic_.value_;
      reward_statistic_.update_from_heuristic_from_backpropagated(heuristic_reward_value);

      const auto heuristic_cost_value = statistic_impl.cost_statistic_.value_;
      cost_statistic_.update_from_heuristic_from_backpropagated(heuristic_cost_value);
    }

    void update_statistic(const NodeStatistic<CostConstrainedStatistic>& changed_child_statistic) {
      const CostConstrainedStatistic& statistic_impl = changed_child_statistic.impl();

      const auto reward_latest_return = statistic_impl.reward_statistic_.latest_return_;
      reward_statistic_.update_statistics_from_backpropagated(reward_latest_return);

      const auto cost_latest_return = statistic_impl.cost_statistic_.latest_return_;
      cost_statistic_.update_statistics_from_backpropagated(cost_latest_return);
    }

    void set_heuristic_estimate(const Reward& accum_rewards, const Cost& accum_ego_cost)
    {
       reward_statistic_.set_heuristic_estimate_from_backpropagated(accum_rewards);
       cost_statistic_.set_heuristic_estimate_from_backpropagated(accum_ego_cost);
    }

    std::string print_node_information() const
    {
        return "";
    }

    std::string print_edge_information(const ActionIdx& action ) const
    {
        return "";
    }

    Reward get_normalized_cost_action_value(const ActionIdx& action) const {
      return cost_statistic_.get_normalized_ucb_value(action);
    }

    MctsParameters make_cost_statistic_parameters(const MctsParameters& mcts_parameters) const {
      MctsParameters cost_statistic_parameters(mcts_parameters);
      cost_statistic_parameters.uct_statistic.EXPLORATION_CONSTANT = mcts_parameters.cost_constrained_statistic.EXPLORATION_CONSTANT;
      cost_statistic_parameters.uct_statistic.LOWER_BOUND = mcts_parameters.cost_constrained_statistic.COST_LOWER_BOUND;
      cost_statistic_parameters.uct_statistic.UPPER_BOUND = mcts_parameters.cost_constrained_statistic.COST_UPPER_BOUND;
      return cost_statistic_parameters;
    }

    MctsParameters make_reward_statistic_parameters(const MctsParameters& mcts_parameters) const {
      MctsParameters reward_statistic_parameters(mcts_parameters);
      reward_statistic_parameters.uct_statistic.EXPLORATION_CONSTANT = mcts_parameters.cost_constrained_statistic.EXPLORATION_CONSTANT;
      reward_statistic_parameters.uct_statistic.LOWER_BOUND = mcts_parameters.cost_constrained_statistic.REWARD_LOWER_BOUND;
      reward_statistic_parameters.uct_statistic.UPPER_BOUND = mcts_parameters.cost_constrained_statistic.REWARD_UPPER_BOUND;
      return reward_statistic_parameters;
    }


private:

    UctStatistic reward_statistic_;
    UctStatistic cost_statistic_;

    const double lambda;
    const double exploration_constant;
    const double action_filter_factor;
    const double cost_constraint;
};

template <>
void NodeStatistic<CostConstrainedStatistic>::update_statistic_parameters(MctsParameters& parameters,
                                            const CostConstrainedStatistic& root_statistic,
                                            const unsigned int& current_iteration) {
  const double current_lambda = parameters.cost_constrained_statistic.LAMBDA;
  const double gradient_update_step =
     parameters.cost_constrained_statistic.GRADIENT_UPDATE_STEP*float(current_iteration-1)/
                                                                float(current_iteration);
  const double cost_constraint = parameters.cost_constrained_statistic.COST_CONSTRAINT;
  const double tau_gradient_clip = parameters.cost_constrained_statistic.TAU_GRADIENT_CLIP;
  const double new_lambda =  CostConstrainedStatistic::calculate_next_lambda(current_lambda,
                                                                            gradient_update_step,
                                                                            cost_constraint,
                                                                            tau_gradient_clip,
                                                                            root_statistic);
  parameters.cost_constrained_statistic.LAMBDA = new_lambda;
  VLOG_EVERY_N(5, int(std::ceil(parameters.MAX_NUMBER_OF_ITERATIONS/100.0))) << "Updated lambda from " << current_lambda << 
    " to " << new_lambda << " in iteration " << current_iteration;
}

} // namespace mcts

#endif