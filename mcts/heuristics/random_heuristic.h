// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================

#ifndef RANDOM_HEURISTIC_H
#define RANDOM_HEURISTIC_H

#include "mcts/mcts.h"
#include <type_traits>
#include <iostream>
#include <chrono>

 namespace mcts {
class CostConstrainedStatistic;
class NeuralCostConstrainedStatistic;

// assumes all agents have equal number of actions and the same node statistic
class RandomHeuristic :  public mcts::Heuristic<RandomHeuristic>, mcts::RandomGenerator
{
public:
    RandomHeuristic(const MctsParameters& mcts_parameters) :
            mcts::Heuristic<RandomHeuristic>(mcts_parameters),
            RandomGenerator(mcts_parameters.RANDOM_SEED) {}

    template<class S, class SE, class SO, class H>
    static std::tuple<Reward,
                      std::unordered_map<AgentIdx, Reward>,
                      EgoCosts,
                      double> rollout(const std::shared_ptr<S> &starting_state,
                        const MctsParameters& mcts_parameters,
                        unsigned current_depth) {      
        namespace chr = std::chrono;
        auto start = std::chrono::high_resolution_clock::now();
        auto state = starting_state->clone();

        Reward ego_accum_reward = 0.0f;
        std::unordered_map<AgentIdx, Reward> other_accum_rewards;
        for (const auto& ai : state->get_other_agent_idx()) {
          other_accum_rewards[ai] = 0.0f;
        }

        EgoCosts accum_cost(state->get_num_costs(), 0.0f);
        const double k_discount_factor = mcts_parameters.DISCOUNT_FACTOR; 
        double modified_discount_factor = k_discount_factor;
        int num_iterations = 0;
        double executed_step_length = 0.0;
        double ego_action_probability = 1.0;
        
        while((!state->is_terminal())&&(num_iterations<mcts_parameters.random_heuristic.MAX_NUMBER_OF_ITERATIONS)&&
                (std::chrono::duration_cast<std::chrono::milliseconds>( std::chrono::high_resolution_clock::now() - start ).count() 
                    < mcts_parameters.random_heuristic.MAX_SEARCH_TIME ) &&
                  current_depth < mcts_parameters.MAX_SEARCH_DEPTH) {
            // Build joint action by calling statistics for each agent
            JointAction jointaction(state->get_num_agents());
            SE ego_statistic(state->get_num_actions(state->get_ego_agent_idx()),
                          state->get_ego_agent_idx(),
                          mcts_parameters);
            jointaction[S::ego_agent_idx] = ego_statistic.choose_next_action(*state);
            AgentIdx action_idx = 1;
            for (const auto& ai : state->get_other_agent_idx()) {
              SO statistic(state->get_num_actions(ai), ai, mcts_parameters);
              jointaction[action_idx] = statistic.choose_next_action(*state);
              action_idx++;
            }

            EgoCosts ego_cost;
            std::vector<Reward> step_rewards(state->get_num_agents());
            auto new_state = state->execute(jointaction, step_rewards, ego_cost);
            modified_discount_factor = k_discount_factor * modified_discount_factor;
            ego_accum_reward += modified_discount_factor*step_rewards[S::ego_agent_idx];
            AgentIdx reward_idx = 1;
            for (const auto& ai : state->get_other_agent_idx()) {
              other_accum_rewards[ai] = modified_discount_factor*step_rewards[reward_idx];
              reward_idx++;
            }

            if(!accum_cost.empty()) {
              for (std::size_t cost_stat_idx = 0; cost_stat_idx < accum_cost.size(); ++cost_stat_idx) {
                // Do not discount costs 
                accum_cost[cost_stat_idx] += ego_cost[cost_stat_idx];
              }
            }
            modified_discount_factor = modified_discount_factor*k_discount_factor;

            executed_step_length += state->get_execution_step_length();
            ego_action_probability *= 1.0/state->get_num_actions(state->get_ego_agent_idx());
            num_iterations +=1;
            current_depth += 1;
            state = new_state->clone();
         };

        // correct estimate by probability that they occcur
        ego_accum_reward *= ego_action_probability;
        if(!accum_cost.empty()) {
            for (std::size_t cost_stat_idx = 0; cost_stat_idx < accum_cost.size(); ++cost_stat_idx) {
              // Do not discount costs 
              accum_cost[cost_stat_idx] *= ego_action_probability;
            }
        }

        return std::make_tuple(ego_accum_reward, other_accum_rewards, accum_cost, executed_step_length);
    }

    template<class S, class SE, class SO, class H>
    std::pair<SE, std::unordered_map<AgentIdx, SO>> calculate_heuristic_values(const std::shared_ptr<StageNode<S,SE,SO,H>> &node) {   
        Reward ego_accum_reward;
        std::unordered_map<AgentIdx, Reward> other_accum_rewards;
        EgoCosts accum_cost;
        double executed_step_length;

        std::tie(ego_accum_reward, other_accum_rewards, accum_cost, executed_step_length) = 
              RandomHeuristic::rollout<S, SE, SO, H>(node->get_state()->clone(), mcts_parameters_, node->get_depth());

        // generate an extra node statistic for each agent
        SE ego_heuristic(0, node->get_state()->get_ego_agent_idx(), mcts_parameters_);
        if constexpr(std::is_same<SE, CostConstrainedStatistic>::value || std::is_same<SE, NeuralCostConstrainedStatistic>::value) {
          ego_heuristic.set_heuristic_estimate(ego_accum_reward, accum_cost, executed_step_length);
          VLOG_EVERY_N(6, 10) << "accum cost = " << accum_cost << ", heuristic_step_length = " << executed_step_length;
         } else {
          ego_heuristic.set_heuristic_estimate(ego_accum_reward, accum_cost); 
        }
        std::unordered_map<AgentIdx, SO> other_heuristic_estimates;
        AgentIdx reward_idx=1;
        for (auto agent_idx : node->get_state()->get_other_agent_idx())
        {
            SO statistic(0, agent_idx, mcts_parameters_);
            statistic.set_heuristic_estimate(other_accum_rewards[agent_idx], accum_cost);
            other_heuristic_estimates.insert(std::pair<AgentIdx, SO>(agent_idx, statistic));
            reward_idx++;
        }
        return std::pair<SE, std::unordered_map<AgentIdx, SO>>(ego_heuristic, other_heuristic_estimates);
    }

};

 } // namespace mcts

#endif