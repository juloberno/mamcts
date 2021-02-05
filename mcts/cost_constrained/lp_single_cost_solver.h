// Copyright (c) 2020 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================

#ifndef SOLVE_LP_SINGLE_COST_H_
#define SOLVE_LP_SINGLE_COST_H_

#include "mcts/statistics/uct_statistic.h"

namespace mcts{

inline PolicySampled lp_single_cost_solver(const std::vector<ActionIdx>& feasible_actions,
           const UctStatistic& cost_statistic, const Cost cost_constraint,
           std::mt19937& random_generator) {
      // Solved for K=1
      const auto& cost_stats = cost_statistic.get_ucb_statistics();

      ActionIdx maximizing_action = feasible_actions.at(0);
      ActionIdx minimizing_action = feasible_actions.at(0);
      for (const auto& feasible_action : feasible_actions ) {
        if(cost_stats.at(feasible_action).action_value_ > cost_stats.at(maximizing_action).action_value_) {
          maximizing_action = feasible_action;
          continue;
        }
        if(cost_stats.at(feasible_action).action_value_ < cost_stats.at(minimizing_action).action_value_) {
          minimizing_action = feasible_action;
          continue;
        }
      }

      Policy stochastic_policy;
      for ( const auto action : cost_stats) {
         stochastic_policy[action.first] = 0.0f;
      }
      if(minimizing_action == maximizing_action) {
        stochastic_policy[minimizing_action] = 1.0f;
        return std::make_pair(minimizing_action, stochastic_policy);
      }

      // Three cases
      const double max_val = cost_stats.at(maximizing_action).action_value_;
      const double min_val = cost_stats.at(minimizing_action).action_value_;
      if( min_val >= cost_constraint) {
          // amin gets probability one, amax gets probability zero
          stochastic_policy[minimizing_action] = 1.0f;
          return std::make_pair(minimizing_action, stochastic_policy);
      } else if( max_val <= cost_constraint) {
         // amax gets probability one, amin gets probability zero
         stochastic_policy[maximizing_action] = 1.0f;
         return std::make_pair(maximizing_action, stochastic_policy);
      } else {
         const double probability_maximizer = (cost_constraint - min_val) / (max_val - min_val);
         std::uniform_real_distribution<double> unif(0, 1);
         stochastic_policy[maximizing_action] = probability_maximizer;
         stochastic_policy[minimizing_action] = 1 - probability_maximizer;
         double sample = unif(random_generator);
         if(sample <= probability_maximizer) {
           return std::make_pair(maximizing_action, stochastic_policy);
         } else {
           return std::make_pair(minimizing_action, stochastic_policy);
         }
      }
    }

}

#endif