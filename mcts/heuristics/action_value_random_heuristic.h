// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================

#ifndef ACTION_VALUE_RANDOM_HEURISTIC_H
#define ACTION_VALUE_RANDOM_HEURISTIC_H

#include "mcts/mcts.h"
#include "mcts/heuristics/random_heuristic.h"
#include <type_traits>
#include <iostream>
#include <chrono>

 namespace mcts {
class CostConstrainedStatistic;

class ActionValueRandomHeuristic :  public mcts::Heuristic<ActionValueRandomHeuristic>, mcts::RandomGenerator
{
public:
    ActionValueRandomHeuristic(const MctsParameters& mcts_parameters) :
            mcts::Heuristic<ActionValueRandomHeuristic>(mcts_parameters),
            RandomGenerator(mcts_parameters.RANDOM_SEED) {}

    template<class S, class SE, class SO, class H>
    std::pair<SE, std::unordered_map<AgentIdx, SO>> calculate_heuristic_values(const std::shared_ptr<StageNode<S,SE,SO,H>> &node) {
        namespace chr = std::chrono;
        auto start = std::chrono::high_resolution_clock::now();
        std::shared_ptr<S> state = node->get_state()->clone();

        // For each ego action do a separate rollout
        auto action_returns = ActionMapping(state->get_num_actions(state->get_ego_agent_idx()), 0.0); 
        auto action_costs = ActionMapping(state->get_num_actions(state->get_ego_agent_idx()), EgoCosts(state->get_num_costs(), 0.0)); 
        auto action_executed_step_lengths = ActionMapping(state->get_num_actions(state->get_ego_agent_idx()), 0.0);  
        auto other_accum_rewards = AgentMapping(state->get_other_agent_idx(), 0.0); 
        for (ActionIdx action_idx = 0; action_idx < state->get_num_actions(state->get_ego_agent_idx())
                                       && !state->is_terminal(); ++action_idx) {
          // Build joint action with specific ego action
          JointAction jointaction(state->get_num_agents());
          jointaction[S::ego_agent_idx] = action_idx;
          AgentIdx agent_idx = 1;
          for (const auto& ai : state->get_other_agent_idx()) {
            SO statistic(state->get_num_actions(ai), ai, mcts_parameters_);
            jointaction[agent_idx] = statistic.choose_next_action(*state);
            agent_idx++;
          }
          EgoCosts ego_cost;
          std::vector<Reward> step_rewards(state->get_num_agents());

          // Execute one step with this joint action
          auto new_state = state->execute(jointaction, step_rewards, ego_cost);

          // Now run a rollout to get value from next state
          Reward ego_accum_reward_rollout;
          std::unordered_map<AgentIdx, Reward> other_accum_rewards_rollout;
          EgoCosts accum_cost_rollout;
          double executed_step_length_rollout;
          std::tie(ego_accum_reward_rollout, other_accum_rewards, 
                    accum_cost_rollout, executed_step_length_rollout) =
                                  RandomHeuristic::rollout<S, SE, SO, H>(new_state, mcts_parameters_, node->get_depth()+1);

          // Accumulate statistics
          action_returns[action_idx] += mcts_parameters_.DISCOUNT_FACTOR*step_rewards[S::ego_agent_idx]
            + mcts_parameters_.DISCOUNT_FACTOR*ego_accum_reward_rollout;
          AgentIdx reward_idx = 1;
          for (const auto& ai : state->get_other_agent_idx()) {
            other_accum_rewards[ai] += mcts_parameters_.DISCOUNT_FACTOR*step_rewards[reward_idx] +
                  mcts_parameters_.DISCOUNT_FACTOR*other_accum_rewards_rollout[ai];
            reward_idx++;
          }

          if(!accum_cost_rollout.empty()) {
            if (action_costs[action_idx].empty()) {
                 action_costs[action_idx] = EgoCosts(accum_cost_rollout.size(), 0.0);
            }
            for (std::size_t cost_stat_idx = 0; cost_stat_idx < accum_cost_rollout.size(); ++cost_stat_idx) {
              // Do not discount costs 
              action_costs[action_idx][cost_stat_idx] += ego_cost[cost_stat_idx] +
                   accum_cost_rollout[cost_stat_idx];
            }
          }
          action_executed_step_lengths[action_idx] += state->get_execution_step_length() + 
                executed_step_length_rollout;
        }

        // generate an extra node statistic for each agent
        SE ego_heuristic(0, node->get_state()->get_ego_agent_idx(), mcts_parameters_);
        if constexpr(std::is_same<SE, CostConstrainedStatistic>::value) {
          ego_heuristic.set_heuristic_estimate(action_returns, action_costs, action_executed_step_lengths);
          //VLOG_EVERY_N(6, 10) << "accum cost = " << action_costs << ", heuristic_step_length = " << action_executed_step_lengths;
          } else {
          ego_heuristic.set_heuristic_estimate(action_returns, action_costs); 
        }
        std::unordered_map<AgentIdx, SO> other_heuristic_estimates;
        AgentIdx reward_idx=1;
        EgoCosts mean_cost(state->get_num_costs(), 0.0);
        for (const auto& action_cost : action_costs) {
          mean_cost += action_cost.second;
        }
        for(auto&  cost : mean_cost) {
          cost /= action_costs.size();
        }
        for (auto agent_idx : node->get_state()->get_other_agent_idx())
        {
            SO statistic(0, agent_idx, mcts_parameters_);
            statistic.set_heuristic_estimate(other_accum_rewards[agent_idx], mean_cost);
            other_heuristic_estimates.insert(std::pair<AgentIdx, SO>(agent_idx, statistic));
            reward_idx++;
        }
        return std::pair<SE, std::unordered_map<AgentIdx, SO>>(ego_heuristic, other_heuristic_estimates);
    }

};

 } // namespace mcts

#endif